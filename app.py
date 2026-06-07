from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path("data/processed/cadenas_unificadas_2025_procesado.csv")
QUARTER_ORDER = ["T1", "T2", "T3", "T4"]


def extraer_tipo_envase(producto: str) -> str:
    producto = str(producto).lower()

    if "botella" in producto:
        return "Botella"
    if "bid\u00f3n" in producto or "bidon" in producto:
        return "Bidon"
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


def clasificar_canasta_basica(producto: str) -> bool:
    producto = str(producto).lower()
    palabras_canasta = [
        "arroz",
        "fideos",
        "pasta",
        "harina",
        "pan",
        "galleta",
        "aceite",
        "azúcar",
        "azucar",
        "sal",
        "yerba",
        "leche",
        "huevo",
        "huevos",
        "queso",
        "manteca",
        "dulce",
        "carne",
        "vacuna",
        "pollo",
        "pescado",
        "merluza",
        "jamón",
        "jamon",
        "chorizo",
        "leonesa",
        "papa",
        "tomate",
        "cebolla",
        "zanahoria",
        "zapallo",
        "banana",
        "manzana",
        "naranja",
        "jabón",
        "jabon",
        "papel higiénico",
        "papel higienico",
        "detergente",
        "hipoclorito",
        "lavandina",
        "alcohol",
    ]
    return any(palabra in producto for palabra in palabras_canasta)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")
    df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")

    text_columns = [
        "Periodo",
        "Grupo",
        "Producto",
        "Super",
        "Rango_precio_producto",
        "Rango_precio",
        "Tipo_envase",
        "Trimestre",
        "Posicion_precio_producto",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    df = df.dropna(subset=["Precio", "Mes", "Producto", "Super", "Grupo"]).reset_index(drop=True)

    if "Tipo_envase" not in df.columns:
        df["Tipo_envase"] = df["Producto"].apply(extraer_tipo_envase)

    if "Es_canasta_basica" not in df.columns:
        df["Es_canasta_basica"] = df["Producto"].apply(clasificar_canasta_basica)
    else:
        df["Es_canasta_basica"] = (
            df["Es_canasta_basica"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["true", "1", "si", "sí", "yes"])
        )

    if "Trimestre" not in df.columns:
        df["Trimestre"] = pd.cut(
            df["Mes"],
            bins=[0, 3, 6, 9, 12],
            labels=QUARTER_ORDER,
        ).astype("string")

    if "Semestre" not in df.columns:
        df["Semestre"] = df["Mes"].apply(lambda value: "S1" if value <= 6 else "S2")

    df["Precio_promedio_producto"] = df.groupby("Producto")["Precio"].transform("mean")
    df["Diferencia_vs_promedio_producto"] = df["Precio"] - df["Precio_promedio_producto"]
    df["Porcentaje_vs_promedio_producto"] = (
        df["Diferencia_vs_promedio_producto"] / df["Precio_promedio_producto"] * 100
    )
    df["Posicion_precio_producto"] = df["Porcentaje_vs_promedio_producto"].apply(
        lambda value: "Mas barato que promedio" if value < 0 else "Mas caro que promedio"
    )
    df["Precio_min_producto"] = df.groupby("Producto")["Precio"].transform("min")
    df["Diferencia_vs_min_producto"] = df["Precio"] - df["Precio_min_producto"]
    df["Ranking_precio_producto"] = df.groupby("Producto")["Precio"].rank(method="dense")
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


def build_filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["Precio"].between(selected_price_range[0], selected_price_range[1])
        & df["Mes"].between(selected_month_range[0], selected_month_range[1])
        & df["Grupo"].isin(selected_groups)
        & df["Super"].isin(selected_supers)
        & df["Tipo_envase"].isin(selected_package_types)
        & df["Trimestre"].isin(selected_quarters)
    ].copy()


def cheapest_rows_by_product(df: pd.DataFrame) -> pd.DataFrame:
    cheapest_idx = df.groupby("Producto")["Precio"].idxmin()
    cheapest = df.loc[
        cheapest_idx,
        [
            "Producto",
            "Grupo",
            "Super",
            "Precio",
            "Periodo",
            "Tipo_envase",
            "Porcentaje_vs_promedio_producto",
        ],
    ].sort_values(["Producto", "Precio"])
    return cheapest.reset_index(drop=True)


def basket_summary(df: pd.DataFrame, selected_products: list[str]) -> pd.DataFrame:
    basket_df = df[df["Producto"].isin(selected_products)].copy()
    if basket_df.empty:
        return pd.DataFrame()

    best_product_super = (
        basket_df.groupby(["Super", "Producto"], as_index=False)["Precio"]
        .min()
        .sort_values(["Super", "Producto", "Precio"])
    )
    summary = (
        best_product_super.groupby("Super")
        .agg(
            Productos_disponibles=("Producto", "nunique"),
            Total_canasta=("Precio", "sum"),
            Precio_promedio=("Precio", "mean"),
        )
        .reset_index()
    )
    summary["Productos_faltantes"] = len(selected_products) - summary["Productos_disponibles"]
    summary["Cobertura_%"] = summary["Productos_disponibles"] / len(selected_products) * 100
    return summary.sort_values(
        ["Productos_faltantes", "Total_canasta", "Precio_promedio"],
        ascending=[True, True, True],
    )


def canasta_promedio_mensual(df: pd.DataFrame) -> pd.DataFrame:
    canasta_df = df[df["Es_canasta_basica"]].copy()
    if canasta_df.empty:
        return pd.DataFrame()

    precio_producto_mes = (
        canasta_df.groupby(["Mes", "Periodo", "Producto"], as_index=False)["Precio"].mean()
    )
    monthly = (
        precio_producto_mes.groupby(["Mes", "Periodo"], as_index=False)["Precio"]
        .sum()
        .rename(columns={"Precio": "Costo_promedio_canasta"})
        .sort_values("Mes")
    )
    monthly_products = (
        precio_producto_mes.groupby(["Mes", "Periodo"], as_index=False)["Producto"]
        .nunique()
        .rename(columns={"Producto": "Productos_canasta"})
    )
    monthly = monthly.merge(monthly_products, on=["Mes", "Periodo"], how="left")
    return monthly


def canasta_super_mes(df: pd.DataFrame) -> pd.DataFrame:
    canasta_df = df[df["Es_canasta_basica"]].copy()
    if canasta_df.empty:
        return pd.DataFrame()

    total_productos = canasta_df["Producto"].nunique()
    precio_producto_super_mes = (
        canasta_df.groupby(["Mes", "Periodo", "Super", "Producto"], as_index=False)["Precio"].mean()
    )
    result = (
        precio_producto_super_mes.groupby(["Mes", "Periodo", "Super"], as_index=False)
        .agg(
            Costo_canasta=("Precio", "sum"),
            Productos_disponibles=("Producto", "nunique"),
        )
        .sort_values(["Mes", "Costo_canasta"])
    )
    result["Total_productos_canasta"] = total_productos
    result["Cobertura_%"] = result["Productos_disponibles"] / total_productos * 100
    return result


def cobertura_super_canasta(df: pd.DataFrame) -> pd.DataFrame:
    canasta_df = df[df["Es_canasta_basica"]].copy()
    if canasta_df.empty:
        return pd.DataFrame()

    total_productos = canasta_df["Producto"].nunique()
    coverage = canasta_df.groupby("Super")["Producto"].nunique().reset_index()
    coverage = coverage.rename(columns={"Producto": "Productos_disponibles"})
    coverage["Total_productos_canasta"] = total_productos
    coverage["Cobertura_%"] = coverage["Productos_disponibles"] / total_productos * 100

    cost = (
        canasta_df.groupby(["Super", "Producto"], as_index=False)["Precio"]
        .mean()
        .groupby("Super", as_index=False)["Precio"]
        .sum()
        .rename(columns={"Precio": "Costo_canasta_promedio"})
    )
    coverage = coverage.merge(cost, on="Super", how="left")
    return coverage.sort_values(
        ["Cobertura_%", "Costo_canasta_promedio"], ascending=[False, True]
    )


st.set_page_config(page_title="Comparador de precios para ahorrar", layout="wide")

df = load_data()

st.title("Comparador de precios para ahorrar")
st.caption(
    "Busca productos, compara supermercados y arma una canasta para decidir donde conviene comprar mas barato."
)

st.sidebar.markdown("## Filtros de compra")
st.sidebar.markdown(
    "Ajusta el periodo, grupos y supermercados. La recomendacion se calcula solo con los datos filtrados."
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
    options=QUARTER_ORDER,
    default=QUARTER_ORDER,
)

filtered_df = build_filtered_data(df)

if filtered_df.empty:
    st.warning("No hay registros para los filtros seleccionados.")
    st.stop()

st.sidebar.markdown("## Buscador de productos")
search_text = st.sidebar.text_input("Buscar por nombre", placeholder="Ej: arroz, aceite, jabon")

product_options = sorted(filtered_df["Producto"].dropna().unique())
if search_text:
    product_options = [
        product for product in product_options if search_text.lower() in product.lower()
    ]

selected_products = st.sidebar.multiselect(
    "Productos para comparar o sumar a la canasta",
    options=product_options,
    default=[],
)

tab_savings, tab_basic_basket, tab_explore, tab_features, tab_data = st.tabs(
    ["Ahorrar en compras", "Canasta basica", "Exploracion", "Variables del EDA", "Datos"]
)

with tab_savings:
    st.subheader("Recomendador de ahorro")
    st.markdown(
        "La app busca el precio mas bajo disponible para cada producto y compara cuanto costaria una canasta en cada supermercado."
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Registros filtrados", f"{len(filtered_df):,}")
    metric_2.metric("Productos disponibles", f"{filtered_df['Producto'].nunique():,}")
    metric_3.metric("Supermercados", f"{filtered_df['Super'].nunique():,}")
    metric_4.metric("Precio medio", f"${filtered_df['Precio'].mean():,.2f}")

    if not selected_products:
        st.info("Selecciona uno o mas productos en el sidebar para ver donde comprarlos mas barato.")

        top_savings = cheapest_rows_by_product(filtered_df).head(25)
        st.markdown("#### Productos con mejor precio encontrado")
        st.dataframe(top_savings, width="stretch")
    else:
        selected_df = filtered_df[filtered_df["Producto"].isin(selected_products)].copy()
        cheapest = cheapest_rows_by_product(selected_df)

        st.markdown("#### Mejor supermercado para cada producto")
        st.dataframe(
            cheapest.rename(
                columns={
                    "Super": "Super mas barato",
                    "Precio": "Precio mas bajo",
                    "Porcentaje_vs_promedio_producto": "% vs promedio del producto",
                }
            ).round(2),
            width="stretch",
        )

        estimated_min_total = cheapest["Precio"].sum()
        avg_total = (
            selected_df.groupby("Producto")["Precio"].mean().reindex(selected_products).sum()
        )
        estimated_saving = avg_total - estimated_min_total

        save_col_1, save_col_2, save_col_3 = st.columns(3)
        save_col_1.metric("Costo minimo combinando supers", f"${estimated_min_total:,.2f}")
        save_col_2.metric("Costo usando precios promedio", f"${avg_total:,.2f}")
        save_col_3.metric("Ahorro estimado", f"${estimated_saving:,.2f}")

        basket = basket_summary(filtered_df, selected_products)
        complete_basket = basket[basket["Productos_faltantes"] == 0].copy()

        st.markdown("#### Ranking de supermercados para tu canasta")
        st.dataframe(basket.round(2), width="stretch")

        if not complete_basket.empty:
            best_super = complete_basket.iloc[0]
            st.success(
                f"Para comprar todo en un solo lugar, conviene ir a {best_super['Super']} "
                f"con una canasta estimada de ${best_super['Total_canasta']:,.2f}."
            )
            if len(complete_basket) > 1:
                worst_complete = complete_basket.iloc[-1]
                one_stop_saving = worst_complete["Total_canasta"] - best_super["Total_canasta"]
                st.caption(
                    f"Elegir el supermercado mas barato frente al mas caro de la lista completa puede ahorrar "
                    f"aproximadamente ${one_stop_saving:,.2f}."
                )
        else:
            best_coverage = basket.iloc[0]
            st.warning(
                f"Ningun supermercado tiene todos los productos filtrados. "
                f"La mejor cobertura la tiene {best_coverage['Super']} con "
                f"{int(best_coverage['Productos_disponibles'])} de {len(selected_products)} productos."
            )

        fig_basket = px.bar(
            basket.head(12),
            x="Super",
            y="Total_canasta",
            color="Productos_faltantes",
            title="Costo estimado de la canasta por supermercado",
            labels={
                "Super": "Supermercado",
                "Total_canasta": "Costo de canasta",
                "Productos_faltantes": "Productos faltantes",
            },
        )
        st.plotly_chart(fig_basket, width="stretch")

        fig_cheapest = px.bar(
            cheapest,
            x="Producto",
            y="Precio",
            color="Super",
            title="Precio mas bajo encontrado por producto",
            labels={"Precio": "Precio mas bajo"},
        )
        st.plotly_chart(fig_cheapest, width="stretch")

    st.markdown("#### Supermercados que tienden a estar mas baratos que el promedio")
    super_position = (
        filtered_df.groupby("Super")["Porcentaje_vs_promedio_producto"]
        .mean()
        .sort_values()
        .reset_index()
    )
    fig_super_position = px.bar(
        super_position,
        x="Super",
        y="Porcentaje_vs_promedio_producto",
        title="Diferencia promedio vs precio promedio del mismo producto",
        labels={
            "Super": "Supermercado",
            "Porcentaje_vs_promedio_producto": "% vs promedio del producto",
        },
    )
    fig_super_position.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_super_position, width="stretch")

with tab_basic_basket:
    st.subheader("Canasta basica familiar estimada")
    st.markdown(
        "Esta seccion usa la variable `Es_canasta_basica` creada en el EDA. "
        "La canasta es una estimacion basada en palabras clave del nombre del producto, "
        "no una canasta oficial. Sirve para comparar cobertura y costo entre supermercados."
    )

    canasta_df = filtered_df[filtered_df["Es_canasta_basica"]].copy()

    if canasta_df.empty:
        st.warning("No hay productos de canasta basica para los filtros seleccionados.")
    else:
        coverage = cobertura_super_canasta(filtered_df)
        monthly = canasta_promedio_mensual(filtered_df)
        super_month = canasta_super_mes(filtered_df)

        basic_col_1, basic_col_2, basic_col_3, basic_col_4 = st.columns(4)
        basic_col_1.metric("Registros de canasta", f"{len(canasta_df):,}")
        basic_col_2.metric("Productos de canasta", f"{canasta_df['Producto'].nunique():,}")
        basic_col_3.metric("Supermercados con canasta", f"{canasta_df['Super'].nunique():,}")
        basic_col_4.metric("Precio medio canasta", f"${canasta_df['Precio'].mean():,.2f}")

        st.markdown("#### Supermercados con mayor cobertura de canasta")
        st.dataframe(coverage.round(2), width="stretch")

        best_coverage = coverage.iloc[0]
        st.success(
            f"{best_coverage['Super']} tiene la mayor cobertura: "
            f"{best_coverage['Cobertura_%']:.1f}% de los productos de la canasta "
            f"({int(best_coverage['Productos_disponibles'])} de "
            f"{int(best_coverage['Total_productos_canasta'])})."
        )

        coverage_col_1, coverage_col_2 = st.columns(2)
        with coverage_col_1:
            fig_coverage = px.bar(
                coverage.head(12),
                x="Super",
                y="Cobertura_%",
                text="Cobertura_%",
                title="Top supermercados por cobertura de canasta",
                labels={"Cobertura_%": "% de productos disponibles"},
            )
            fig_coverage.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_coverage.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_coverage, width="stretch")

        with coverage_col_2:
            fig_coverage_cost = px.scatter(
                coverage,
                x="Cobertura_%",
                y="Costo_canasta_promedio",
                size="Productos_disponibles",
                color="Super",
                hover_data=["Productos_disponibles", "Total_productos_canasta"],
                title="Cobertura vs costo estimado de canasta",
                labels={
                    "Cobertura_%": "% de cobertura",
                    "Costo_canasta_promedio": "Costo estimado de canasta",
                },
            )
            st.plotly_chart(fig_coverage_cost, width="stretch")

        st.markdown("#### Costo promedio mensual de la canasta")
        st.dataframe(monthly.round(2), width="stretch")

        fig_monthly = px.line(
            monthly,
            x="Periodo",
            y="Costo_promedio_canasta",
            markers=True,
            title="Costo promedio estimado de canasta basica por mes",
            labels={"Costo_promedio_canasta": "Costo promedio de canasta"},
        )
        st.plotly_chart(fig_monthly, width="stretch")

        st.markdown("#### Costo de canasta por supermercado y mes")
        min_coverage = st.slider(
            "Cobertura minima para comparar costo por supermercado",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
        )
        comparable_super_month = super_month[super_month["Cobertura_%"] >= min_coverage].copy()

        if comparable_super_month.empty:
            st.warning("No hay supermercados con esa cobertura minima en los filtros actuales.")
        else:
            latest_month = int(comparable_super_month["Mes"].max())
            latest_ranking = comparable_super_month[
                comparable_super_month["Mes"] == latest_month
            ].sort_values(["Costo_canasta", "Cobertura_%"], ascending=[True, False])

            st.markdown("##### Ranking del ultimo mes filtrado")
            st.dataframe(latest_ranking.round(2), width="stretch")

            best_cost = latest_ranking.iloc[0]
            st.info(
                f"Para el ultimo mes filtrado ({best_cost['Periodo']}), "
                f"el menor costo con al menos {min_coverage}% de cobertura es "
                f"{best_cost['Super']} con ${best_cost['Costo_canasta']:,.2f} "
                f"y {best_cost['Cobertura_%']:.1f}% de cobertura."
            )

            fig_super_month = px.line(
                comparable_super_month,
                x="Periodo",
                y="Costo_canasta",
                color="Super",
                markers=True,
                title=f"Costo de canasta por supermercado y mes (cobertura >= {min_coverage}%)",
                labels={"Costo_canasta": "Costo de canasta"},
            )
            st.plotly_chart(fig_super_month, width="stretch")

            fig_latest = px.bar(
                latest_ranking.head(12),
                x="Super",
                y="Costo_canasta",
                color="Cobertura_%",
                title="Supermercados mas convenientes en el ultimo mes filtrado",
                labels={"Costo_canasta": "Costo de canasta"},
            )
            st.plotly_chart(fig_latest, width="stretch")

with tab_explore:
    st.subheader("Analisis descriptivo interactivo")

    summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
    summary_col_1.metric("Registros", f"{len(filtered_df):,}")
    summary_col_2.metric("Productos", f"{filtered_df['Producto'].nunique():,}")
    summary_col_3.metric("Tipos de envase", f"{filtered_df['Tipo_envase'].nunique():,}")
    summary_col_4.metric("Supers por producto prom.", f"{filtered_df['Cantidad_supers_producto'].mean():,.1f}")

    st.markdown("#### Resumen descriptivo")
    st.dataframe(descriptive_summary(filtered_df).round(2), width="stretch")

    hist_col, scatter_col = st.columns(2)
    with hist_col:
        fig_hist = px.histogram(
            filtered_df,
            x="Precio",
            nbins=40,
            color="Grupo",
            title="Distribucion de precios",
            labels={"Precio": "Precio", "count": "Cantidad"},
        )
        fig_hist.update_layout(bargap=0.05)
        st.plotly_chart(fig_hist, width="stretch")

    with scatter_col:
        numeric_columns = filtered_df.select_dtypes(include="number").columns.tolist()
        x_default = numeric_columns.index("Mes") if "Mes" in numeric_columns else 0
        y_default = numeric_columns.index("Precio") if "Precio" in numeric_columns else 0
        x_axis = st.selectbox("Variable X", options=numeric_columns, index=x_default)
        y_axis = st.selectbox("Variable Y", options=numeric_columns, index=y_default)
        fig_scatter = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color="Grupo",
            hover_data=["Periodo", "Producto", "Super"],
            title=f"Relacion entre {x_axis} y {y_axis}",
            opacity=0.65,
        )
        st.plotly_chart(fig_scatter, width="stretch")

    group_summary = (
        filtered_df.groupby("Grupo")["Precio"]
        .agg(Registros="count", Media="mean", Mediana="median", Minimo="min", Maximo="max")
        .sort_values("Media", ascending=False)
    )
    st.markdown("#### Precio por grupo")
    st.dataframe(group_summary.round(2), width="stretch")

    fig_group = px.bar(
        group_summary.reset_index(),
        x="Grupo",
        y="Media",
        title="Precio medio por grupo",
        labels={"Media": "Precio medio"},
    )
    st.plotly_chart(fig_group, width="stretch")

with tab_features:
    st.subheader("Variables creadas para entender ahorro")

    feature_col_1, feature_col_2 = st.columns(2)
    with feature_col_1:
        package_counts = filtered_df["Tipo_envase"].value_counts().reset_index()
        package_counts.columns = ["Tipo_envase", "Registros"]
        fig_package_counts = px.bar(
            package_counts,
            x="Tipo_envase",
            y="Registros",
            title="Cantidad de registros por tipo de envase",
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
            labels={"Precio": "Precio promedio"},
        )
        st.plotly_chart(fig_package_price, width="stretch")

    feature_col_3, feature_col_4 = st.columns(2)
    with feature_col_3:
        quarter_price = (
            filtered_df.groupby("Trimestre", observed=False)["Precio"]
            .mean()
            .reindex(QUARTER_ORDER)
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
        fig_pct_dist = px.histogram(
            filtered_df,
            x="Porcentaje_vs_promedio_producto",
            nbins=40,
            color="Grupo",
            title="Distribucion del % vs promedio del producto",
        )
        st.plotly_chart(fig_pct_dist, width="stretch")

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
        labels={"Diferencia": "Maximo - minimo"},
    )
    st.plotly_chart(fig_spread, width="stretch")

with tab_data:
    st.subheader("Datos filtrados")
    st.dataframe(filtered_df.head(500), width="stretch")
    st.download_button(
        "Descargar datos filtrados",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="precios_filtrados.csv",
        mime="text/csv",
    )

st.info("El dataset no contiene columnas de latitud y longitud, por lo que no se incluye mapa geografico.")
