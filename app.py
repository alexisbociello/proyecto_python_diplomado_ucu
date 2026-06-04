from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path("data/processed/cadenas_unificadas_2025_procesado.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")
    df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")

    text_columns = ["Periodo", "Grupo", "Producto", "Super", "Rango_precio_producto", "Rango_precio"]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df.dropna(subset=["Precio", "Mes"]).reset_index(drop=True)


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    summary = pd.DataFrame(
        {
            "Media": numeric_df.mean(),
            "Mediana": numeric_df.median(),
            "Desv. estandar": numeric_df.std(),
            "Minimo": numeric_df.min(),
            "Q1": numeric_df.quantile(0.25),
            "Q2": numeric_df.quantile(0.50),
            "Q3": numeric_df.quantile(0.75),
            "Maximo": numeric_df.max(),
        }
    )
    summary["Rango"] = summary["Maximo"] - summary["Minimo"]
    return summary[
        ["Media", "Mediana", "Desv. estandar", "Minimo", "Q1", "Q2", "Q3", "Maximo", "Rango"]
    ]


st.set_page_config(
    page_title="Analisis de precios 2025",
    layout="wide",
)

df = load_data()

st.title("Analisis interactivo de precios 2025")
st.caption("Exploracion descriptiva del dataset de precios por producto, supermercado y periodo.")

st.sidebar.markdown("## Controles de analisis")
st.sidebar.markdown(
    "Usa los filtros para explorar el dataset. Los resumenes y graficos se actualizan automaticamente."
)

price_min = float(df["Precio"].min())
price_max = float(df["Precio"].max())
selected_price_range = st.sidebar.slider(
    "Rango de precio",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
)

month_min = int(df["Mes"].min())
month_max = int(df["Mes"].max())
selected_month_range = st.sidebar.slider(
    "Rango de mes",
    min_value=month_min,
    max_value=month_max,
    value=(month_min, month_max),
)

selected_groups = st.sidebar.multiselect(
    "Grupo",
    options=sorted(df["Grupo"].dropna().unique()),
    default=sorted(df["Grupo"].dropna().unique()),
)

selected_supers = st.sidebar.multiselect(
    "Supermercado",
    options=sorted(df["Super"].dropna().unique()),
    default=sorted(df["Super"].dropna().unique()),
)

filtered_df = df[
    df["Precio"].between(selected_price_range[0], selected_price_range[1])
    & df["Mes"].between(selected_month_range[0], selected_month_range[1])
    & df["Grupo"].isin(selected_groups)
    & df["Super"].isin(selected_supers)
].copy()

st.subheader("Datos filtrados")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Registros", f"{len(filtered_df):,}")
metric_2.metric("Productos", f"{filtered_df['Producto'].nunique():,}")
metric_3.metric("Supermercados", f"{filtered_df['Super'].nunique():,}")
metric_4.metric("Precio medio", f"${filtered_df['Precio'].mean():,.2f}" if not filtered_df.empty else "-")

if filtered_df.empty:
    st.warning("No hay registros para los filtros seleccionados.")
    st.stop()

with st.expander("Ver muestra del DataFrame filtrado", expanded=False):
    st.dataframe(filtered_df.head(200), width="stretch")

st.subheader("Resumen descriptivo")
st.dataframe(descriptive_summary(filtered_df).round(2), width="stretch")

st.subheader("Visualizacion dinamica")

hist_col, scatter_col = st.columns(2)

with hist_col:
    fig_hist = px.histogram(
        filtered_df,
        x="Precio",
        nbins=40,
        color="Grupo",
        title="Distribucion del target: Precio",
        labels={"Precio": "Precio", "count": "Cantidad de registros", "Grupo": "Grupo"},
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, width="stretch")

with scatter_col:
    numeric_columns = filtered_df.select_dtypes(include="number").columns.tolist()
    x_axis = st.selectbox("Variable X del scatter", options=numeric_columns, index=numeric_columns.index("Mes"))
    y_axis = st.selectbox("Variable Y del scatter", options=numeric_columns, index=numeric_columns.index("Precio"))

    fig_scatter = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color="Grupo",
        hover_data=["Periodo", "Producto", "Super"],
        title=f"Relacion entre {x_axis} y {y_axis}",
        labels={x_axis: x_axis, y_axis: y_axis, "Grupo": "Grupo"},
        opacity=0.65,
    )
    st.plotly_chart(fig_scatter, width="stretch")

st.subheader("Analisis por categoria")

group_summary = (
    filtered_df.groupby("Grupo")["Precio"]
    .agg(Registros="count", Media="mean", Mediana="median", Minimo="min", Maximo="max")
    .sort_values("Media", ascending=False)
)
st.dataframe(group_summary.round(2), width="stretch")

fig_group = px.bar(
    group_summary.reset_index(),
    x="Grupo",
    y="Media",
    title="Precio medio por grupo",
    labels={"Media": "Precio medio", "Grupo": "Grupo"},
)
st.plotly_chart(fig_group, width="stretch")

st.info("El dataset no contiene columnas de latitud y longitud, por lo que no se incluye mapa geografico.")
