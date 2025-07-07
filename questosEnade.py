import streamlit as st
import requests
import textwrap
import json
from datetime import datetime
from io import BytesIO
from bs4 import BeautifulSoup
from openai import OpenAI
import PyPDF2

# --- 1. CONFIGURAÇÃO DA PÁGINA & API KEY ---
st.set_page_config(page_title="Gerador de Questões ENADE", page_icon="🎓", layout="wide")
st.sidebar.header("🔑 Configuração da API")
api_key = st.sidebar.text_input("Chave OpenAI", type="password")
model = st.sidebar.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-3.5-turbo"])
if not api_key:
    st.sidebar.warning("Informe sua chave da OpenAI para continuar.")
    st.stop()

# --- 2. DEFINIÇÃO DO ESCOPO ---
st.header("1. Escopo da Questão")
AREAS = {
    "Ciências Sociais Aplicadas": ["Administração", "Direito", "Comunicação Social"],
    "Engenharias": ["Engenharia de Software", "Engenharia Civil", "Engenharia Elétrica"],
    "Ciências da Saúde": ["Medicina", "Enfermagem", "Farmácia"]
}
area = st.selectbox("Grande Área", list(AREAS.keys()))
curso = st.selectbox("Curso", AREAS[area])
assunto = st.text_input("Tópico / Assunto central", placeholder="Ex: IA na arbitragem")

# --- 3. CARREGAR TEXTO-BASE (URL ou PDF) ---
st.header("2. Texto-Base (situação-estímulo)")
metodo = st.radio("Origem do texto-base:", ["URL", "PDF"], horizontal=True)
if metodo == "URL":
    url = st.text_input("Cole a URL completa")
    if st.button("▶️ Extrair de URL"):
        try:
            r = requests.get(url, timeout=10); r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","header","footer","nav","aside"]):
                tag.decompose()
            st.session_state.texto_fonte = " ".join(soup.stripped_strings)
            st.session_state.fonte_link = url
        except Exception as e:
            st.error(f"Falha ao extrair URL: {e}")
elif metodo == "PDF":
    pdf = st.file_uploader("Envie um PDF", type="pdf")
    if pdf:
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf.read()))
            st.session_state.texto_fonte = "".join(p.extract_text() or "" for p in reader.pages)
            st.session_state.fonte_link = pdf.name
        except Exception as e:
            st.error(f"Falha ao ler PDF: {e}")

if st.session_state.get("texto_fonte"):
    st.success("✔ Texto-base carregado!")
    with st.expander("Ver / editar texto-base"):
        st.session_state.texto_fonte = st.text_area(
            "Texto-Fonte", st.session_state.texto_fonte, height=300
        )

# --- 4. TRECHO-BASE: SELEÇÃO OU RESUMO ---
st.header("3. Trecho-Base")
if st.session_state.get("texto_fonte"):
    modo = st.radio("Como obter o trecho-base?", ["Selecionar parágrafo(s)", "Resumo automático"], horizontal=True)
    if modo.startswith("Selecionar"):
        paras = [p.strip() for p in st.session_state.texto_fonte.split("\n") if len(p.strip()) > 80]
        sel = st.multiselect(
            "Escolha um ou mais parágrafos:",
            paras,
            format_func=lambda p: textwrap.shorten(p, 120, placeholder="…")
        )
        if sel:
            st.session_state.trecho = "\n\n".join(sel)
    else:
        if st.button("🔎 Resumir texto completo"):
            client = OpenAI(api_key=api_key)
            prompt = (
                "Resuma em até 3 frases este texto para servir de base "
                "a uma situação-problema ENADE:\n\n"
                + st.session_state.texto_fonte
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.5,
                max_tokens=200
            )
            st.session_state.trecho = resp.choices[0].message.content.strip()
        if st.session_state.get("trecho"):
            st.session_state.trecho = st.text_area(
                "Resumo (edite se quiser)", st.session_state.trecho, height=150
            )

# --- 5. CONTEXTO (situação-problema) ---
st.header("4. Contexto da Questão")
if st.session_state.get("trecho"):
    if not st.session_state.get("contexto"):
        client = OpenAI(api_key=api_key)
        prompt = (
            "Com base neste trecho, gere UMA BREVE situação-problema "
            "profissional e relevante para uma questão ENADE. Retorne apenas o texto:\n\n"
            + st.session_state.trecho
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=300
        )
        st.session_state.contexto = resp.choices[0].message.content.strip()
    st.session_state.contexto = st.text_area(
        "Edite o contexto se necessário:", st.session_state.contexto, height=120
    )

# --- 6. REFERÊNCIA ABNT (prefilled e editável) ---
st.header("5. Referência ABNT")
if st.session_state.get("fonte_link"):
    # gera referência automática
    hoje = datetime.now()
    meses = ["jan.","fev.","mar.","abr.","mai.","jun.","jul.","ago.","set.","out.","nov.","dez."]
    acesso = f"{hoje.day} {meses[hoje.month-1]} {hoje.year}"
    referencia_default = (
        f"{st.session_state.fonte_link}. "
        f"Disponível em: <{st.session_state.fonte_link}>. "
        f"Acesso em: {acesso}."
    )
    # campo editável
    st.session_state.referencia = st.text_area(
        "Referência ABNT (edite se quiser):",
        value=referencia_default,
        height=100
    )

# --- 7. FORMULÁRIO E GERAÇÃO DA QUESTÃO ENADE ---
st.header("6. Gerar Questão ENADE")
if st.session_state.get("contexto") and st.session_state.get("trecho") and st.session_state.get("referencia"):
    with st.form("enade_form"):
        tipo   = st.selectbox("Tipo de item", ["Múltipla Escolha", "Asserção-Razão", "Discursivo"])
        perfil = st.text_input("Perfil do egresso", placeholder="Ex: crítico e reflexivo")
        comp   = st.text_input("Competência", placeholder="Ex: analisar conflitos éticos")
        obj    = st.text_input("Objeto de conhecimento", placeholder="Ex: ética profissional")
        diff   = st.select_slider("Dificuldade", ["Fácil","Média","Difícil"], value="Média")
        extra  = st.text_area("Info. adicional (opcional)")
        submit = st.form_submit_button("🚀 Gerar Questão")
    if submit:
        system_prompt = """
Você é docente especialista INEP. Crie uma questão padrão ENADE, seguindo rigorosamente:
- Originalidade total
- Texto-base indispensável e referenciado
- Enunciado afirmativo e claro
- 5 alternativas A–E, 1 correta
- Distratores plausíveis
- Linguagem formal, norma-padrão
- Foco em aplicação (situação-problema)
- Evitar termos absolutos (sempre, nunca, apenas, etc.)
- Ao final indique "Gabarito: Letra X"
- Inclua justificativas breves para cada alternativa
"""
        user_prompt = f"""
Contexto:
{st.session_state.contexto}

Texto-Base:
{st.session_state.trecho}

Referência (ABNT):
{st.session_state.referencia}

Encomenda:
- Curso: {curso}
- Assunto: {assunto}
- Tipo de item: {tipo}
- Perfil do egresso: {perfil}
- Competência: {comp}
- Objeto de conhecimento: {obj}
- Dificuldade: {diff}
- Observações: {extra}
"""
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        raw = resp.choices[0].message.content
        try:
            st.session_state.questao = json.loads(raw)
        except:
            st.session_state.questao = raw

# --- 8. EXIBIÇÃO & DOWNLOAD ---
st.header("7. Resultado")
q = st.session_state.get("questao")
if q:
    if isinstance(q, dict):
        st.json(q)
        st.download_button(
            "📥 Baixar JSON",
            data=json.dumps(q, ensure_ascii=False, indent=2),
            file_name="questao_enade.json"
        )
    else:
        st.markdown(q)
