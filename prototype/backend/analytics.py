"""
Analytics Engine
=================
Computes real conversion attribution and campaign performance metrics
from the provided datasets. No simulated KPIs — everything is computed
from actual POS, WhatsApp, digital funnel, and grower scan data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from backend.data_loader import get_data_store


class AnalyticsEngine:
    """Computes campaign performance and conversion attribution from real data."""

    def __init__(self):
        self.store = get_data_store()

    def get_whatsapp_funnel_metrics(self) -> Dict[str, Any]:
        """Compute real open/click/delivery rates from WhatsApp log."""
        wa = self.store.whatsapp_log.copy()

        total = len(wa)
        delivered = wa["delivered_status"].apply(lambda x: str(x).strip().lower() == "true").sum()
        opened = wa["opened_status"].apply(lambda x: str(x).strip().lower() == "true").sum()
        clicked = wa["clicked_status"].apply(lambda x: str(x).strip().lower() == "true").sum()

        # By product
        by_product = {}
        for product in wa["campaign_product"].unique():
            subset = wa[wa["campaign_product"] == product]
            n = len(subset)
            d = subset["delivered_status"].apply(lambda x: str(x).strip().lower() == "true").sum()
            o = subset["opened_status"].apply(lambda x: str(x).strip().lower() == "true").sum()
            c = subset["clicked_status"].apply(lambda x: str(x).strip().lower() == "true").sum()
            by_product[product] = {
                "sent": int(n),
                "delivered": int(d),
                "opened": int(o),
                "clicked": int(c),
                "delivery_rate": round(d / n, 4) if n > 0 else 0,
                "open_rate": round(o / d, 4) if d > 0 else 0,
                "click_rate": round(c / o, 4) if o > 0 else 0,
                "ctr_overall": round(c / n, 4) if n > 0 else 0,
            }

        return {
            "total_messages": int(total),
            "delivered": int(delivered),
            "opened": int(opened),
            "clicked": int(clicked),
            "delivery_rate": round(delivered / total, 4) if total else 0,
            "open_rate": round(opened / delivered, 4) if delivered else 0,
            "click_rate": round(clicked / opened, 4) if opened else 0,
            "by_product": by_product,
        }

    def get_digital_funnel_metrics(self) -> Dict[str, Any]:
        """Aggregate digital funnel performance by campaign."""
        df = self.store.digital_funnel.copy()

        campaigns = {}
        for cid in df["campaign_id"].unique():
            subset = df[df["campaign_id"] == cid]
            crop = subset["campaign_crop"].iloc[0]
            product = subset["campaign_product"].iloc[0]
            total_imps = int(subset["social_post_impression"].sum())
            total_visits = int(subset["landing_page_visits"].sum())
            total_leads = int(subset["lead_form_submission"].sum())

            # Weekly trend
            weekly_trend = []
            for _, row in subset.sort_values("week_start_date").iterrows():
                weekly_trend.append({
                    "week": row["week_start_date"].strftime("%Y-%m-%d"),
                    "impressions": int(row["social_post_impression"]),
                    "visits": int(row["landing_page_visits"]),
                    "leads": int(row["lead_form_submission"]),
                })

            campaigns[cid] = {
                "campaign_crop": crop,
                "campaign_product": product,
                "total_impressions": total_imps,
                "total_visits": total_visits,
                "total_leads": total_leads,
                "impression_to_visit_rate": round(total_visits / total_imps, 4) if total_imps else 0,
                "visit_to_lead_rate": round(total_leads / total_visits, 4) if total_visits else 0,
                "overall_conversion": round(total_leads / total_imps, 6) if total_imps else 0,
                "weeks": len(subset),
                "weekly_trend": weekly_trend,
            }

        return {"campaigns": campaigns}

    def get_conversion_attribution(self) -> Dict[str, Any]:
        """
        Attribution: correlate WhatsApp campaigns with downstream product scans and POS.
        This is the core Campaign-to-Action metric.
        """
        store = self.store
        wa = store.whatsapp_log.copy()
        growers = store.growers.copy()

        # Join WhatsApp messages with grower product scans
        wa_grower = wa.merge(
            growers[["grower_id", "product_scan", "product_name", "product_scan_datetime",
                      "tehsil", "crop"]],
            on="grower_id", how="left"
        )

        # Convert dates to check attribution window
        wa_grower["msg_date"] = pd.to_datetime(wa_grower["message_sent_date"])
        wa_grower["scan_date"] = pd.to_datetime(wa_grower["product_scan_datetime"], errors="coerce")

        # Clicked AND scanned within 14 days AFTER the message
        wa_grower["clicked"] = wa_grower["clicked_status"].apply(
            lambda x: str(x).strip().lower() == "true"
        )
        
        # Valid scan condition: happened, and occurred between msg_date and msg_date + 14 days
        # Use pandas isnull() to safely handle NaT
        wa_grower["valid_scan"] = (
            wa_grower["product_scan"].apply(lambda x: str(x).strip().lower() == "true") & 
            ~wa_grower["scan_date"].isnull() &
            (wa_grower["scan_date"] >= wa_grower["msg_date"]) &
            (wa_grower["scan_date"] <= wa_grower["msg_date"] + timedelta(days=14))
        )

        total_messages = len(wa_grower)
        total_clicked = int(wa_grower["clicked"].sum())
        total_scanned_after_msg = int((wa_grower["clicked"] & wa_grower["valid_scan"]).sum())

        # Campaign-to-action rate
        cta_rate = round(total_scanned_after_msg / total_messages, 4) if total_messages else 0
        click_to_scan = round(total_scanned_after_msg / total_clicked, 4) if total_clicked else 0

        # By crop
        by_crop = {}
        for crop in wa_grower["campaign_crop"].unique():
            subset = wa_grower[wa_grower["campaign_crop"] == crop]
            n = len(subset)
            cl = int(subset["clicked"].sum())
            sc = int((subset["clicked"] & subset["valid_scan"]).sum())
            by_crop[crop] = {
                "messages": int(n),
                "clicked": cl,
                "scanned_after_click": sc,
                "campaign_to_action_rate": round(sc / n, 4) if n else 0,
            }

        return {
            "total_messages": total_messages,
            "total_clicked": total_clicked,
            "total_converted_scan": total_scanned_after_msg,
            "campaign_to_action_rate": cta_rate,
            "click_to_scan_rate": click_to_scan,
            "by_crop": by_crop,
        }

    def get_pos_trends(self, sku_name: str = None, state: str = None) -> Dict[str, Any]:
        """Get POS sales trends by week, optionally filtered."""
        pos = self.store.retailer_pos.copy()

        if sku_name:
            pos = pos[pos["sku_name"] == sku_name]
        if state:
            # Join with retailers to get state
            retailers = self.store.retailers[["retailer_id", "state"]]
            pos = pos.merge(retailers, on="retailer_id", how="left")
            pos = pos[pos["state"] == state]

        pos["week"] = pos["transaction_date"].dt.to_period("W").apply(lambda x: x.start_time)
        weekly = pos.groupby("week").agg(
            total_qty=("sku_qty", "sum"),
            total_revenue=("sku_price", lambda x: (x * pos.loc[x.index, "sku_qty"]).sum()),
            transaction_count=("transaction_id", "nunique"),
        ).reset_index()

        weekly_data = []
        for _, row in weekly.sort_values("week").iterrows():
            weekly_data.append({
                "week": row["week"].strftime("%Y-%m-%d"),
                "total_qty": int(row["total_qty"]),
                "total_revenue": round(float(row["total_revenue"]), 2),
                "transactions": int(row["transaction_count"]),
            })

        return {
            "filter_sku": sku_name,
            "filter_state": state,
            "total_weeks": len(weekly_data),
            "weekly_data": weekly_data,
        }

    def get_inventory_health(self) -> Dict[str, Any]:
        """Compute current inventory health: out-of-stock rates by product and region."""
        inv = self.store.retailer_inventory.copy()
        latest_week = inv["week_end_date"].max()
        current = inv[inv["week_end_date"] == latest_week]

        # Join with retailer locations
        retailers = self.store.retailers[["retailer_id", "state", "district", "tehsil"]]
        current = current.merge(retailers, on="retailer_id", how="left")

        # By SKU
        by_sku = {}
        for sku in current["sku_name"].unique():
            subset = current[current["sku_name"] == sku]
            total_retailers = len(subset)
            out_of_stock = int((subset["sku_qty"] == 0).sum())
            avg_stock = round(subset["sku_qty"].mean(), 1)
            by_sku[sku] = {
                "retailers_carrying": total_retailers,
                "out_of_stock_count": out_of_stock,
                "out_of_stock_rate": round(out_of_stock / total_retailers, 4) if total_retailers else 0,
                "avg_stock_qty": avg_stock,
            }

        # By state
        by_state = {}
        for state in current["state"].unique():
            subset = current[current["state"] == state]
            total = len(subset)
            oos = int((subset["sku_qty"] == 0).sum())
            by_state[state] = {
                "total_sku_retailer_combos": total,
                "out_of_stock": oos,
                "oos_rate": round(oos / total, 4) if total else 0,
            }

        return {
            "snapshot_week": latest_week.strftime("%Y-%m-%d"),
            "by_sku": by_sku,
            "by_state": by_state,
        }

    def get_field_activity_summary(self) -> Dict[str, Any]:
        """Summarize rep field activities from visit logs."""
        visits = self.store.retailer_visits.copy()

        by_type = visits["visit_type"].value_counts().to_dict()
        by_product = visits["product_recommended"].value_counts().to_dict()

        # Monthly trend
        visits["month"] = visits["visit_date"].dt.to_period("M").apply(lambda x: x.start_time)
        monthly = visits.groupby("month").size().reset_index(name="visit_count")
        monthly_data = [
            {"month": row["month"].strftime("%Y-%m"), "visits": int(row["visit_count"])}
            for _, row in monthly.sort_values("month").iterrows()
        ]

        return {
            "total_visits": len(visits),
            "by_type": {k: int(v) for k, v in by_type.items()},
            "by_product": {k: int(v) for k, v in by_product.items()},
            "monthly_trend": monthly_data,
        }
