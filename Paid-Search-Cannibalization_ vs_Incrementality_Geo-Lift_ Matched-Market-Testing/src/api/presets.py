"""Three one-click hospitality presets shared by the Streamlit app and the e2e test."""

PRESETS = {
    "A": {"label": "Miami Beach flagship (Tier 1, heavy OTA conquesting)", "tier": 1, "organic_rank": 2.6, "ota_pressure": 0.75,
          "competitor_adr": 300.0, "loyalty_pen": 0.28, "lead_time": 45.0, "baseline_direct_bookings_day": 58.0,
          "ota_to_direct_ratio": 0.85, "lastclick_share": 0.32, "cannibalization_rate": 0.55, "ota_capture_rate": 0.45,
          "diversion_rate": 0.45, "adr": 285.0, "commission_rate": 0.20, "cpc": 3.2},
    "B": {"label": "Nashville regional (Tier 2, near break-even)", "tier": 2, "organic_rank": 2.2, "ota_pressure": 0.35,
          "competitor_adr": 215.0, "loyalty_pen": 0.24, "lead_time": 28.0, "baseline_direct_bookings_day": 25.0,
          "ota_to_direct_ratio": 0.55, "lastclick_share": 0.30, "cannibalization_rate": 0.82, "ota_capture_rate": 0.20,
          "diversion_rate": 0.45, "adr": 205.0, "commission_rate": 0.175, "cpc": 2.2},
    "C": {"label": "Asheville drive-to (Tier 3, top organic rank, little OTA pressure)", "tier": 3, "organic_rank": 1.3,
          "ota_pressure": 0.12, "competitor_adr": 170.0, "loyalty_pen": 0.26, "lead_time": 14.0,
          "baseline_direct_bookings_day": 10.0, "ota_to_direct_ratio": 0.42, "lastclick_share": 0.30,
          "cannibalization_rate": 0.93, "ota_capture_rate": 0.06, "diversion_rate": 0.45, "adr": 165.0,
          "commission_rate": 0.15, "cpc": 1.5},
}
