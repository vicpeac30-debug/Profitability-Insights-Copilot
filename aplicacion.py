import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Credit Card FP&A Copilot",
    page_icon="💳",
    layout="wide"
)

@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent
    customer = pd.read_csv(base_path / "customer_master.csv")
    cc = pd.read_csv(base_path / "cc_customer_month.csv")
    pricing = pd.read_csv(base_path / "pricing_events_month.csv")

    customer["origination_month"] = pd.to_datetime(customer["origination_month"])
    cc["month_id"] = pd.to_datetime(cc["month_id"])
    pricing["month_id"] = pd.to_datetime(pricing["month_id"])

    df = cc.merge(customer, on="customer_id", how="left")
    df = df.merge(pricing, on=["customer_id", "month_id"], how="left")

    # Homologar segmentos
    df["segment"] = (
        df["segment"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    segment_mapping = {
        "MASIVO": "MASS",
        "MASS": "MASS",
        "AFFLUENT": "AFFLUENT",
        "EMERGING": "EMERGING"
    }

    df["segment"] = df["segment"].replace(segment_mapping)

    # Homologar subsegmentos
    df["subsegment"] = (
        df["subsegment"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["fee_income"] = (
        df["annual_fee"]
        + df["late_fee"]
        + df["overlimit_fee"]
        + df["installment_fee"]
    )

    df["interest_income"] = df["income_interest"]
    df["interchange_income"] = df["interchange"]

    df["total_income"] = (
        df["fee_income"]
        + df["interest_income"]
        + df["interchange_income"]
    )

    df["total_cost"] = (
        df["rewards_cost"]
        + df["funding_cost"]
        + df["servicing_cost"]
        + df["credit_cost"]
    )

    df["margin_before_waiver"] = df["total_income"] - df["total_cost"]
    df["fee_waiver_abs"] = df["fee_waiver"].abs()

    df["year"] = df["month_id"].dt.year
    df["month"] = df["month_id"].dt.month

    return df


df = load_data()

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Filtros")

year_filter = st.sidebar.multiselect(
    "Año",
    options=[2024, 2025],
    default=[2024, 2025]
)

month_names = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre"
}

month_filter = st.sidebar.multiselect(
    "Mes",
    options=list(month_names.keys()),
    default=list(month_names.keys()),
    format_func=lambda x: month_names[x]
)

segment_filter = st.sidebar.multiselect(
    "Segmento",
    sorted(df["segment"].dropna().unique())
)

product_filter = st.sidebar.multiselect(
    "Producto",
    sorted(df["product_tier"].dropna().unique())
)

income_filter = st.sidebar.multiselect(
    "Nivel de ingreso",
    sorted(df["income_band"].dropna().unique())
)

risk_filter = st.sidebar.multiselect(
    "Riesgo",
    sorted(df["risk_band"].dropna().unique())
)

df_filtered = df[
    (df["year"].isin(year_filter)) &
    (df["month"].isin(month_filter))
].copy()

if segment_filter:
    df_filtered = df_filtered[df_filtered["segment"].isin(segment_filter)]

if product_filter:
    df_filtered = df_filtered[df_filtered["product_tier"].isin(product_filter)]

if income_filter:
    df_filtered = df_filtered[df_filtered["income_band"].isin(income_filter)]

if risk_filter:
    df_filtered = df_filtered[df_filtered["risk_band"].isin(risk_filter)]


# =========================
# TITLE
# =========================

st.title("Credit Card FP&A Copilot")
st.caption("Agente analítico para explicar rentabilidad, ingresos, atrición, fee waivers y drivers del negocio.")


# =========================
# KPIS
# =========================

total_profit = df_filtered["profit"].sum()
fee_income = df_filtered["fee_income"].sum()
attrition_rate = df_filtered["attrition_flag"].mean()
fee_waivers = df_filtered["fee_waiver_abs"].sum()

customer_profit = (
    df_filtered.groupby("customer_id")["profit"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

if len(customer_profit) > 0:
    top20_n = max(int(len(customer_profit) * 0.20), 1)
    top20_share = (
        customer_profit.head(top20_n)["profit"].sum()
        / customer_profit["profit"].sum()
    )
else:
    top20_share = 0

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Rentabilidad total", f"${total_profit:,.0f}")
col2.metric("Fee Income", f"${fee_income:,.0f}")
col3.metric("Attrition Rate", f"{attrition_rate:.2%}")
col4.metric("Fee Waivers", f"${fee_waivers:,.0f}")
col5.metric("Profit Top 20%", f"{top20_share:.1%}")


# =========================
# MONTHLY DATA
# =========================

monthly = (
    df_filtered.groupby("month_id")
    .agg(
        customers=("customer_id", "nunique"),
        profit=("profit", "sum"),
        fee_income=("fee_income", "sum"),
        interest_income=("interest_income", "sum"),
        interchange_income=("interchange_income", "sum"),
        fee_waiver_abs=("fee_waiver_abs", "sum"),
        attritions=("attrition_flag", "sum"),
        attrition_rate=("attrition_flag", "mean"),
        campaign_contact=("campaign_contact", "sum"),
        campaign_response=("campaign_response", "sum"),
        retention_offer=("retention_offer", "sum"),
        pricing_exception=("pricing_exception", "sum")
    )
    .reset_index()
)


# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Resumen Ejecutivo",
    "Drivers",
    "Segmentos",
    "Copilot",
    "Recomendaciones"
])


with tab1:
    st.subheader("Diagnóstico ejecutivo")

    st.markdown("""
    ### Conclusión principal

    El deterioro de rentabilidad no proviene de una caída en ingresos.  
    Los ingresos por comisiones, intereses e interchange continúan creciendo; sin embargo,
    a partir de junio-julio 2025 se observa una caída relevante en profit acompañada por
    mayor atrición y mayores fee waivers.
    """)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(monthly, x="month_id", y="profit", title="Profit mensual")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(monthly, x="month_id", y="attrition_rate", title="Attrition Rate")
        st.plotly_chart(fig, use_container_width=True)

    income_long = monthly.melt(
        id_vars="month_id",
        value_vars=["fee_income", "interest_income", "interchange_income"],
        var_name="Tipo de ingreso",
        value_name="Monto"
    )

    fig = px.line(
        income_long,
        x="month_id",
        y="Monto",
        color="Tipo de ingreso",
        title="Mix de ingresos"
    )
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.subheader("Drivers del deterioro")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            monthly,
            x="month_id",
            y="fee_waiver_abs",
            title="Fee Waivers"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pricing_long = monthly.melt(
            id_vars="month_id",
            value_vars=[
                "campaign_contact",
                "campaign_response",
                "retention_offer",
                "pricing_exception"
            ],
            var_name="Acción",
            value_name="Cantidad"
        )

        fig = px.line(
            pricing_long,
            x="month_id",
            y="Cantidad",
            color="Acción",
            title="Acciones comerciales y pricing"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Lectura ejecutiva

    Durante el segundo semestre de 2025 aumentaron las campañas, ofertas de retención,
    respuestas de campaña, excepciones de pricing y fee waivers.  
    Esto sugiere una estrategia de retención principalmente reactiva ante el incremento de atrición.
    """)


with tab3:
    st.subheader("Rentabilidad por segmento, producto, ingreso y riesgo")

    col1, col2 = st.columns(2)

    with col1:
        segment_profit = (
            df_filtered.groupby("segment")["profit"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            segment_profit,
            x="segment",
            y="profit",
            title="Profit promedio por segmento"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tier_profit = (
            df_filtered.groupby("product_tier")["profit"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            tier_profit,
            x="product_tier",
            y="profit",
            title="Profit promedio por producto"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        income_profit = (
            df_filtered.groupby("income_band")["profit"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            income_profit,
            x="income_band",
            y="profit",
            title="Profit promedio por ingreso"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_profit = (
            df_filtered.groupby("risk_band")["profit"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            risk_profit,
            x="risk_band",
            y="profit",
            title="Profit promedio por riesgo"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pareto de rentabilidad")

    customer_profit["cum_profit_pct"] = (
        customer_profit["profit"].cumsum()
        / customer_profit["profit"].sum()
    )

    customer_profit["customer_pct"] = (
        np.arange(1, len(customer_profit) + 1)
        / len(customer_profit)
    )

    fig = px.line(
        customer_profit,
        x="customer_pct",
        y="cum_profit_pct",
        title="Concentración de profit por cliente"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    ### Insight

    El top 20% de clientes genera aproximadamente **{top20_share:.1%}** de la rentabilidad total.
    """)


with tab4:
    st.subheader("Pregunta al FP&A Copilot")

    suggested_questions = [
        "¿Por qué cayó la rentabilidad?",
        "¿Los ingresos por comisiones están cayendo?",
        "¿Qué cambió en julio 2025?",
        "¿Qué segmentos son más rentables?",
        "¿Qué productos generan mayor rentabilidad?",
        "¿Qué clientes están abandonando?",
        "¿Las campañas de retención están funcionando?",
        "¿Qué impacto tienen los fee waivers?",
        "¿Qué significa el Pareto de rentabilidad?",
        "¿Qué recomendaciones darías?"
    ]

    selected_question = st.selectbox(
        "Preguntas sugeridas",
        [""] + suggested_questions
    )

    question = st.text_input(
        "O escribe tu propia pregunta",
        value=selected_question
    )

    def copilot_answer(q):
        q = q.lower()

        if "rentabilidad" in q or "profit" in q or "utilidad" in q:
            return f"""
            La rentabilidad se deteriora principalmente a partir de junio-julio 2025.
            La caída no se explica por menores ingresos, ya que fee income, intereses
            e interchange continúan creciendo. El principal riesgo está en la pérdida
            de clientes de alto valor y el incremento de fee waivers.

            El top 20% de clientes genera aproximadamente {top20_share:.1%}
            de la utilidad total.
            """

        elif "comision" in q or "fee income" in q:
            return """
            Los ingresos por comisiones no muestran una caída estructural.
            Durante 2025 mantienen una tendencia creciente. El problema está más asociado
            a rentabilidad neta, fee waivers y pérdida de clientes rentables.
            """

        elif "julio" in q or "2025" in q:
            return """
            A partir de junio-julio 2025 se observa un punto de inflexión:
            cae el profit, aumenta la atrición, crecen los fee waivers y se intensifican
            las acciones comerciales como campaign contacts, retention offers y pricing exceptions.
            """

        elif "segmento" in q:
            return """
            El segmento más rentable es AFFLUENT. MASS y EMERGING tienen menor profit promedio.
            Por ello, la estrategia de retención debe priorizar clientes AFFLUENT.
            """

        elif "producto" in q:
            return """
            Los productos más rentables son Black y Platinum. Estos productos concentran
            clientes de mayor valor y deben ser priorizados en las estrategias comerciales.
            """

        elif "campaña" in q or "retención" in q or "retencion" in q or "pricing" in q:
            return """
            Las campañas, ofertas de retención y pricing exceptions aumentan en el segundo semestre de 2025.
            Esto sugiere que el banco reaccionó ante el incremento de atrición, pero la estrategia debe
            evolucionar hacia un enfoque predictivo y basado en valor económico.
            """

        elif "waiver" in q or "condonacion" in q:
            return """
            Los fee waivers aumentan de forma importante durante 2025.
            Pueden ser útiles como herramienta de retención, pero deben aplicarse de forma selectiva
            considerando profit esperado, riesgo de abandono y valor del cliente.
            """

        elif "pareto" in q or "top 20" in q:
            return f"""
            El Pareto muestra que el top 20% de clientes genera aproximadamente {top20_share:.1%}
            de la rentabilidad total. Esto confirma que el portafolio tiene alta concentración
            de valor y que perder clientes premium tiene un impacto desproporcionado.
            """

        elif "recomend" in q:
            return """
            Las recomendaciones principales son:
            1. Priorizar clientes AFFLUENT, Black y Platinum.
            2. Revisar fee waivers y pricing exceptions.
            3. Implementar alertas tempranas de atrición.
            4. Crear modelos predictivos de churn.
            5. Incorporar Customer Lifetime Value.
            """

        else:
            return """
            Puedo responder preguntas sobre profit, fee income, attrition rate,
            fee waivers, segmentos, productos, pricing exceptions, campañas y recomendaciones.
            """

    if question:
        st.markdown("### Respuesta del Copilot")
        st.write(copilot_answer(question))


with tab5:
    st.subheader("Recomendaciones ejecutivas")

    st.markdown("""
    ### Corto plazo

    - Priorizar clientes AFFLUENT, Black y Platinum.
    - Revisar políticas de fee waivers.
    - Monitorear attrition de clientes rentables.
    - Implementar alertas tempranas para clientes de alto valor.

    ### Mediano plazo

    - Desarrollar modelos predictivos de atrición.
    - Incorporar Customer Lifetime Value.
    - Optimizar campañas de retención.
    - Evaluar efectividad de pricing exceptions.

    ### Largo plazo

    - Implementar un framework de rentabilidad por cliente.
    - Crear un motor de recomendaciones comerciales.
    - Convertir el análisis en una capacidad continua de Analytics FP&A.
    """)

    st.subheader("Controles del AI Copilot")

    st.markdown("""
    - Validación de calidad de datos.
    - Homologación de segmentos.
    - Cálculos trazables.
    - Revisión humana para decisiones de pricing.
    - No automatizar fee waivers sin aprobación del negocio.
    """)
