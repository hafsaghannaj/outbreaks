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


def risk_label(score: float) -> str:
    if score >= 80:
        return "Very High"
    if score >= 65:
        return "High"
    if score >= 50:
        return "Elevated"
    if score >= 35:
        return "Moderate"
    return "Low"


def add_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 22px; left: 22px; z-index: 9999;
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 12px 14px;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        font-size: 13px; line-height: 1.5;
        ">
      <div style="font-weight: 800; letter-spacing: .2px; margin-bottom: 8px;">Risk Score</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:darkred;border-radius:999px;margin-right:8px;"></span>80–100 • Very High</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:red;border-radius:999px;margin-right:8px;"></span>65–79 • High</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:orange;border-radius:999px;margin-right:8px;"></span>50–64 • Elevated</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:blue;border-radius:999px;margin-right:8px;"></span>35–49 • Moderate</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:green;border-radius:999px;margin-right:8px;"></span>0–34 • Low</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def add_status_box(m: folium.Map, total_points: int, high_points: int, high_threshold: float, shown_min: float, shown_max: float) -> None:
    status_html = f"""
    <div style="
        position: fixed;
        bottom: 22px; right: 22px; z-index: 9999;
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 12px 14px;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        font-size: 13px; line-height: 1.5;
        max-width: 320px;
        ">
      <div style="font-weight: 800; letter-spacing: .2px; margin-bottom: 8px;">Map Status</div>
      <div><b>Total points:</b> {total_points}</div>
      <div><b>High-risk (≥ {high_threshold:.0f}):</b> {high_points}</div>
      <div><b>Risk range:</b> {shown_min:.1f} – {shown_max:.1f}</div>
      <div style="margin-top:8px; color:#3b3b3b;">Use layer toggle (top-right) to switch views.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(status_html))


def add_hotspots_panel(m: folium.Map, top_df: pd.DataFrame) -> None:
    items_html = ""
    for i, row in enumerate(top_df.itertuples(index=False), start=1):
        score = float(row.predicted_risk_score)
        label = risk_label(score)
        color = risk_color(score)
        lat = float(row.lat)
        lon = float(row.lon)
        dt = str(row.date)

        badge = f"""
        <span style="
          display:inline-block; padding:2px 10px; border-radius:999px;
          background:{color}; color:white; font-weight:800; font-size:11px;
          ">
          {label}
        </span>
        """

        items_html += f"""
        <div
          onclick="window.__hb_flyTo({lat}, {lon});"
          style="
            cursor:pointer;
            padding:10px 10px;
            border-radius:14px;
            border:1px solid rgba(0,0,0,0.06);
            background: rgba(255,255,255,0.72);
            margin-top:10px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
            transition: transform .12s ease, box-shadow .12s ease;
          "
          onmouseenter="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 10px 22px rgba(0,0,0,0.12)';"
          onmouseleave="this.style.transform='translateY(0px)'; this.style.boxShadow='0 6px 16px rgba(0,0,0,0.08)';"
          >
          <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
            <div style="font-weight:900; font-size:13px;">#{i} • {score:.1f}</div>
            {badge}
          </div>
          <div style="margin-top:6px; font-size:12px; color:#303030;">
            <div>{dt}</div>
            <div style="opacity:0.85;">{lat:.3f}, {lon:.3f}</div>
          </div>
        </div>
        """

    panel_html = f"""
    <style>
      /* Keep Leaflet control area clear (top-right). */
      .hb-panel {{
        position: fixed;
        top: 70px; left: 70px; z-index: 9999;
        width: 280px;
        max-height: calc(100vh - 220px);
        overflow: auto;
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 14px 14px;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.14);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      }}
      .hb-panel::-webkit-scrollbar {{ width: 8px; }}
      .hb-panel::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.18); border-radius: 999px; }}
      .hb-panel-collapsed {{
        width: 168px;
        max-height: 52px;
        overflow: hidden;
      }}
      .hb-title {{
        font-weight: 950;
        letter-spacing: .2px;
        font-size: 18px;
      }}
      .hb-sub {{
        margin-top: 4px;
        font-size: 12px;
        color: #222;
        opacity: .85;
      }}
      .hb-chip {{
        font-weight: 950;
        font-size: 11px;
        letter-spacing: .6px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(0,0,0,0.06);
        white-space: nowrap;
      }}
      .hb-toggle {{
        cursor: pointer;
        border: 1px solid rgba(0,0,0,0.10);
        background: rgba(255,255,255,0.7);
        border-radius: 999px;
        padding: 6px 10px;
        font-weight: 900;
        font-size: 11px;
      }}
      @media (max-width: 700px) {{
        .hb-panel {{ width: 260px; top: 86px; }}
      }}
    </style>

    <div id="hbPanel" class="hb-panel hb-panel-collapsed">
      <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:10px;">
        <div>
          <div class="hb-title">Outbreaks</div>
          <div class="hb-sub">AI + Satellite Early Warning</div>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px; align-items:flex-end;">
          <div class="hb-chip">HOTSPOTS</div>
          <div id="hbToggle" class="hb-toggle">OPEN</div>
        </div>
      </div>

      <div id="hbBody" style="margin-top:12px;">
        <div style="font-size:12px; color:#2a2a2a;">
          Top 10 predicted risk points (click to zoom).
        </div>
        {items_html}
        <div style="margin-top:14px; font-size:11px; opacity:.65;">
          Outbreaks • Early Warning Demo
        </div>
      </div>
    </div>

    <script>
      window.__hb_flyTo = function(lat, lon) {{
        try {{
          var mapObj = null;
          for (var k in window) {{
            if (k.startsWith('map_') && window[k] && window[k].setView) {{
              mapObj = window[k];
              break;
            }}
          }}
          if (!mapObj) return;
          mapObj.setView([lat, lon], 9, {{animate:true}});
        }} catch (e) {{}}
      }};

      (function() {{
        var panel = document.getElementById('hbPanel');
        var toggle = document.getElementById('hbToggle');
        var body = document.getElementById('hbBody');
        var open = false;

        function setState(nextOpen) {{
          open = nextOpen;
          if (open) {{
            panel.classList.remove('hb-panel-collapsed');
            toggle.textContent = 'CLOSE';
            body.style.display = 'block';
          }} else {{
            panel.classList.add('hb-panel-collapsed');
            toggle.textContent = 'OPEN';
            body.style.display = 'none';
          }}
        }}

        toggle.addEventListener('click', function(e) {{
          e.preventDefault();
          setState(!open);
        }});

        // Start collapsed to avoid overlap, user can open.
        setState(false);
      }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(panel_html))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Folium risk map from scored points.")
    p.add_argument("--high-threshold", type=float, default=65.0, help="Threshold for the High-risk layer")
    return p.parse_args()


def add_points_layer(m: folium.Map, df: pd.DataFrame, name: str) -> None:
    fg = folium.FeatureGroup(name=name, show=(name == "All points"))
    cluster = MarkerCluster(name=f"{name} cluster").add_to(fg)

    for row in df.itertuples(index=False):
        lat = float(row.lat)
        lon = float(row.lon)
        score = float(row.predicted_risk_score)
        dt = str(row.date)

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=risk_color(score),
            fill=True,
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>Risk:</b> {score:.1f}<br><b>Date:</b> {dt}<br><b>Coords:</b> {lat:.3f}, {lon:.3f}",
                max_width=300,
            ),
        ).add_to(cluster)

    fg.add_to(m)


def main() -> None:
    args = parse_args()
    df_all = pd.read_csv(RESULTS_DIR / "risk_scored_points.csv")

    df_high = df_all[df_all["predicted_risk_score"] >= args.high_threshold].copy()
    top10 = df_all.sort_values("predicted_risk_score", ascending=False).head(10).copy()

    if df_all.empty:
        raise SystemExit("No points to plot.")

    m = folium.Map(
        location=[float(df_all["lat"].mean()), float(df_all["lon"].mean())],
        zoom_start=6,
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer("OpenStreetMap", name="Street", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        show=False,
    ).add_to(m)

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

    add_hotspots_panel(m, top10)

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = RESULTS_DIR / "risk_map.html"
    m.save(str(out_path))
    print("Wrote:", out_path)
    print("All points:", len(df_all))
    print("High-risk points:", len(df_high))


if __name__ == "__main__":
    main()
