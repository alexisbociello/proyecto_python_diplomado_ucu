from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path("data/processed/cadenas_unificadas_2025_procesado.csv")


def extraer_tipo_envase(producto: str) -> str:
    producto = str(producto).lower()

    if "botella" in producto:
        return "Botella"
    if "bidón" in producto or "bidon" in producto:
        return "Bidón"
    if "paquete" in producto:
        return "Paquete"
    if "envase" in producto:
        return "Envase"
    if "lata" in producto:
        return "Lata"
    if "sachet" in producto:
        return "Sachet"
    if "caja" in producto:
        return "Caja"
    if "bolsa" in producto:
        return "Bolsa"
    if "tetrabrick" in producto or "tetra" in producto:
        return "Tetrabrick"
    if "frasco" in producto:
        return "Frasco"
    if "pote" in producto:
        return "Pote"
    if "aerosol" in producto:
        return "Aerosol"
    if "spray" in producto:
        return "Spray"
    if "rollo" in producto or "rollos" in producto:
        return "Rollo"
    if "unidad" in producto or "un." in producto or " us." in producto or " us" in producto:
        return "Unidad"
    if any(unit in producto for unit in ["kg", "gr", "grs", "ml", "cm3", "lts", "lt."]):
        return "Al peso"
    return "Sin identificar"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")
    df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")

    text_columns = ["Periodo", "Grupo", "Producto", "Super", "Rango_precio_producto", "Rango_precio"]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    df = df.dropna(subset=["Precio", "Mes", "Producto", "Super", "Grupo"]).reset_index(drop=True)

    df["Tipo_envase"] = df["Producto"].apply(extraer_tipo_envase)
    df["Trimestre"] = pd.cut(
        df["Mes"],
        bins=[0, 3, 6, 9, 12],
        labels=["T1", "T2", "T3", "T4"],
    ).astype("string")
    df["Precio_promedio_producto"] = df.groupby("Producto")["Precio"].transform("mean")
    df["Diferencia_vs_promedio_producto"] = df["Precio"] - df["Precio_promedio_producto"]
    df["Porcentaje_vs_promedio_producto"] = (
        df["Diferencia_vs_promedio_producto"] / df["Precio_promedio_producto"] * 100
    )
    df["Posicion_precio_producto"] = df["Porcentaje_vs_promedio_producto"].apply(
        lambda value: "Mas barato que promedio" if value < 0 else "Mas caro que promedio"
    )
    df["Cantidad_supers_producto"] = df.groupby("Producto")["Super"].transform("nunique")

    return df.reset_index(drop=True)


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

selected_package_types = st.sidebar.multiselect(
    "Tipo de envase",
    options=sorted(df["Tipo_envase"].dropna().unique()),
    default=sorted(df["Tipo_envase"].dropna().unique()),
)

selected_quarters = st.sidebar.multiselect(
    "Trimestre",
    options=["T1", "T2", "T3", "T4"],
    default=["T1", "T2", "T3", "T4"],
)

selected_price_position = st.sidebar.multiselect(
    "Posicion vs promedio del producto",
    options=sorted(df["Posicion_precio_producto"].dropna().unique()),
    default=sorted(df["Posicion_precio_producto"].dropna().unique()),
)

filtered_df = df[
    df["Precio"].between(selected_price_range[0], selected_price_range[1])
    & df["Mes"].between(selected_month_range[0], selected_month_range[1])
    & df["Grupo"].isin(selected_groups)
    & df["Super"].isin(selected_supers)
    & df["Tipo_envase"].isin(selected_package_types)
    & df["Trimestre"].isin(selected_quarters)
    & df["Posicion_precio_producto"].isin(selected_price_position)
].copy()

st.subheader("Datos filtrados")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Registros", f"{len(filtered_df):,}")
metric_2.metric("Productos", f"{filtered_df['Producto'].nunique():,}")
metric_3.metric("Supermercados", f"{filtered_df['Super'].nunique():,}")
metric_4.metric("Precio medio", f"${filtered_df['Precio'].mean():,.2f}" if not filtered_df.empty else "-")

metric_5, metric_6, metric_7, metric_8 = st.columns(4)
metric_5.metric("Tipos de envase", f"{filtered_df['Tipo_envase'].nunique():,}")
metric_6.metric("Supers por producto promedio", f"{filtered_df['Cantidad_supers_producto'].mean():,.1f}")
metric_7.metric("Diferencia media vs producto", f"${filtered_df['Diferencia_vs_promedio_producto'].mean():,.2f}")
metric_8.metric("% medio vs producto", f"{filtered_df['Porcentaje_vs_promedio_producto'].mean():,.2f}%")

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

st.subheader("Variables creadas en el EDA")

feature_col_1, feature_col_2 = st.columns(2)

with feature_col_1:
    package_counts = filtered_df["Tipo_envase"].value_counts().reset_index()
    package_counts.columns = ["Tipo_envase", "Registros"]
    fig_package_counts = px.bar(
        package_counts,
        x="Tipo_envase",
        y="Registros",
        title="Cantidad de registros por tipo de envase",
        labels={"Tipo_envase": "Tipo de envase", "Registros": "Cantidad de registros"},
    )
    st.plotly_chart(fig_package_counts, width="stretch")

with feature_col_2:
    package_price = (
        filtered_df.groupby("Tipo_envase")["Precio"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_package_price = px.bar(
        package_price,
        x="Tipo_envase",
        y="Precio",
        title="Precio promedio por tipo de envase",
        labels={"Tipo_envase": "Tipo de envase", "Precio": "Precio promedio"},
    )
    st.plotly_chart(fig_package_price, width="stretch")

feature_col_3, feature_col_4 = st.columns(2)

with feature_col_3:
    quarter_price = (
        filtered_df.groupby("Trimestre", observed=False)["Precio"]
        .mean()
        .reindex(["T1", "T2", "T3", "T4"])
        .dropna()
        .reset_index()
    )
    fig_quarter = px.bar(
        quarter_price,
        x="Trimestre",
        y="Precio",
        title="Precio promedio por trimestre",
        labels={"Precio": "Precio promedio"},
    )
    st.plotly_chart(fig_quarter, width="stretch")

with feature_col_4:
    super_vs_product = (
        filtered_df.groupby("Super")["Porcentaje_vs_promedio_producto"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_super_vs_product = px.bar(
        super_vs_product,
        x="Super",
        y="Porcentaje_vs_promedio_producto",
        title="Diferencia porcentual promedio vs producto",
        labels={
            "Super": "Supermercado",
            "Porcentaje_vs_promedio_producto": "% vs promedio del producto",
        },
    )
    fig_super_vs_product.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_super_vs_product, width="stretch")

feature_col_5, feature_col_6 = st.columns(2)

with feature_col_5:
    fig_pct_dist = px.histogram(
        filtered_df,
        x="Porcentaje_vs_promedio_producto",
        nbins=40,
        color="Grupo",
        title="Distribucion del % vs promedio del producto",
        labels={"Porcentaje_vs_promedio_producto": "% vs promedio del producto"},
    )
    st.plotly_chart(fig_pct_dist, width="stretch")

with feature_col_6:
    product_price_spread = (
        filtered_df.groupby("Producto")["Precio"]
        .agg(Minimo="min", Maximo="max")
        .assign(Diferencia=lambda data: data["Maximo"] - data["Minimo"])
        .sort_values("Diferencia", ascending=False)
        .head(15)
        .reset_index()
    )
    fig_spread = px.bar(
        product_price_spread,
        x="Producto",
        y="Diferencia",
        title="Top 15 productos con mayor diferencia de precio",
        labels={"Producto": "Producto", "Diferencia": "Maximo - minimo"},
    )
    st.plotly_chart(fig_spread, width="stretch")

st.info("El dataset no contiene columnas de latitud y longitud, por lo que no se incluye mapa geografico.")
