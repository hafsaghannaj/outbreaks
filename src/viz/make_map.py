from __future__ import annotations

import argparse
import pandas as pd
import folium
from folium.plugins import MarkerCluster

from src.config.settings import RESULTS_DIR


def risk_color(score: float) -> str:
    if score >= 80:
        return "darkred"
    if score >= 65:
        return "red"
    if score >= 50:
        return "orange"
    if score >= 35:
        return "blue"
    return "green"


def add_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; z-index: 9999;
        background: white; padding: 10px 12px;
        border: 1px solid #ccc; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-size: 14px; line-height: 1.4;
        ">
      <div style="font-weight: 700; margin-bottom: 6px;">Risk Score Legend</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:darkred;border-radius:50%;margin-right:8px;"></span>80–100 (Very High)</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:red;border-radius:50%;margin-right:8px;"></span>65–79 (High)</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:orange;border-radius:50%;margin-right:8px;"></span>50–64 (Elevated)</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:blue;border-radius:50%;margin-right:8px;"></span>35–49 (Moderate)</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:green;border-radius:50%;margin-right:8px;"></span>0–34 (Low)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def add_status_box(m: folium.Map, total_points: int, high_points: int, high_threshold: float, shown_min: float, shown_max: float) -> None:
    status_html = f"""
    <div style="
        position: fixed;
        bottom: 30px; right: 30px; z-index: 9999;
        background: white; padding: 10px 12px;
        border: 1px solid #ccc; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-size: 13px; line-height: 1.4;
        max-width: 280px;
        ">
      <div style="font-weight: 700; margin-bottom: 6px;">Map Status</div>
      <div><b>Total points:</b> {total_points}</div>
      <div><b>High-risk points (≥ {high_threshold:.0f}):</b> {high_points}</div>
      <div><b>Risk range (all):</b> {shown_min:.1f} – {shown_max:.1f}</div>
      <div style="margin-top:6px; color:#444;">Use the layer toggle (top-right) to switch views.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(status_html))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Folium risk map from scored points.")
    p.add_argument("--high-threshold", type=float, default=65.0, help="Threshold for the High-risk layer")
    return p.parse_args()


def add_points_layer(m: folium.Map, df: pd.DataFrame, name: str) -> None:
    fg = folium.FeatureGroup(name=name, show=(name == "All points"))
    cluster = MarkerCluster(name=f"{name} cluster").add_to(fg)

    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        score = float(row["predicted_risk_score"])
        dt = str(row["date"])

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=risk_color(score),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"Risk: {score:.1f}<br>Date: {dt}<br>Lat/Lon: {lat:.3f}, {lon:.3f}",
                max_width=300,
            ),
        ).add_to(cluster)

    fg.add_to(m)


def main() -> None:
    args = parse_args()
    df_all = pd.read_csv(RESULTS_DIR / "risk_scored_points.csv")

    df_high = df_all[df_all["predicted_risk_score"] >= args.high_threshold].copy()

    if df_all.empty:
        raise SystemExit("No points to plot.")

    m = folium.Map(
        location=[float(df_all["lat"].mean()), float(df_all["lon"].mean())],
        zoom_start=6,
        control_scale=True,
    )

    add_legend(m)
    add_status_box(
        m,
        total_points=len(df_all),
        high_points=len(df_high),
        high_threshold=args.high_threshold,
        shown_min=float(df_all["predicted_risk_score"].min()),
        shown_max=float(df_all["predicted_risk_score"].max()),
    )

    add_points_layer(m, df_all, "All points")
    if len(df_high) > 0:
        add_points_layer(m, df_high, f"High risk (≥ {args.high_threshold:.0f})")

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = RESULTS_DIR / "risk_map.html"
    m.save(str(out_path))
    print("Wrote:", out_path)
    print("All points:", len(df_all))
    print("High-risk points:", len(df_high))


if __name__ == "__main__":
    main()
