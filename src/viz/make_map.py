from __future__ import annotations

import pandas as pd
import folium
from folium.plugins import MarkerCluster

from src.config.settings import RESULTS_DIR


def risk_color(score: float) -> str:
    # Simple 5-band coloring
    if score >= 80:
        return "darkred"
    if score >= 65:
        return "red"
    if score >= 50:
        return "orange"
    if score >= 35:
        return "blue"
    return "green"


def main() -> None:
    scored_path = RESULTS_DIR / "risk_scored_points.csv"
    df = pd.read_csv(scored_path)

    # Center map on mean lat/lon
    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True)
    cluster = MarkerCluster(name="Risk Points").add_to(m)

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
            popup=folium.Popup(f"Risk: {score:.1f}<br>Date: {dt}<br>Lat/Lon: {lat:.3f}, {lon:.3f}", max_width=300),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)

    out_path = RESULTS_DIR / "risk_map.html"
    m.save(str(out_path))

    print("Wrote:", out_path)
    print("Open it in your browser to view the map.")


if __name__ == "__main__":
    main()
