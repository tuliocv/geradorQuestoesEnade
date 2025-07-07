import streamlit as st
import requests
import textwrap
import json
from datetime import datetime
from io import BytesIO
from bs4 import BeautifulSoup
from openai import OpenAI
import PyPDF2

# --- Taxonomia de Bloom ---
BLOOM_LEVELS = ["Lembrar", "Compreender", "Aplicar", "Analisar", "Avaliar", "Criar"]
BLOOM_VERBS = {
    "Lembrar":      ["definir", "listar", "identificar", "recordar", "nomear"],
    "Compreender":  ["explicar", "resumir", "interpretar", "classificar", "descrever"],
    "Aplicar":      ["usar", "implementar", "executar", "demonstrar", "resolver"],
    "Analisar":     ["diferenciar", "organizar", "atribuir", "comparar", "examinar"],
    "Avaliar":      ["julgar", "criticar", "justificar", "avaliar", "defender"],
    "Criar":        ["projetar", "construir", "formular", "sintetizar", "planejar"]
}

# --- 1. CONFIGURAÇÃO DA PÁGINA & API KEY ---
st.set_page_config(page_title="Gerador de Questões ENADE", page_icon="🎓", layout="wide")
st.sidebar.header("🔑 Configuração da API")
api_key = st.sidebar.text_input("Chave OpenAI", type="password")
model = st.sidebar.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-3.5-turbo"])
if not api_key:
    st.sidebar.warning("Informe sua chave da OpenAI para continuar.")
    st.stop()

# --- Funções auxiliares ---
@st.cache_data(ttl=3600)
def extrair_texto_url(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","header","footer","nav","aside"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        st.error(f"Erro ao extrair URL: {e}")
        return None

@st.cache_data
def extrair_texto_pdf(upload) -> str | None:
    try:
        reader = PyPDF2.PdfReader(BytesIO(upload.read()))
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Erro ao ler PDF: {e}")
        return None

def gerar_questao_llm(system_prompt: str, user_prompt: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return resp.choices[0].message.content

def gerar_llm(prompt: str, role_system: str="user", temperature=0.7, max_tokens=300) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": role_system, "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- 2. Escopo ---
st.header("1. Escopo da Questão")
area    = st.text_input("Área do conhecimento", placeholder="Ex: Engenharias")
curso   = st.text_input("Curso",               placeholder="Ex: Engenharia de Software")
assunto = st.text_input("Tópico / Assunto central", placeholder="Ex: IA na arbitragem")

# --- 3. Carregar Texto-Base ---
st.header("2. Texto-Base (situação-estímulo)")
metodo = st.radio("Origem do texto-base:", ["URL", "PDF"], horizontal=True)
if metodo == "URL":
    url = st.text_input("Cole a URL completa")
    if st.button("▶️ Extrair de URL"):
        txt = extrair_texto_url(url)
        if txt:
            st.session_state.texto_fonte = txt
            st.session_state.fonte_link = url
elif metodo == "PDF":
    pdf = st.file_uploader("Envie um PDF", type="pdf")
    if pdf:
        txt = extrair_texto_pdf(pdf)
        if txt:
            st.session_state.texto_fonte = txt
            st.session_state.fonte_link = pdf.name

if st.session_state.get("texto_fonte"):
    with st.expander("Ver / editar texto-base"):
        st.session_state.texto_fonte = st.text_area(
            "Texto-Fonte", st.session_state.texto_fonte, height=300
        )

# --- 4. Trecho-Base ---
st.header("3. Trecho-Base")
if st.session_state.get("texto_fonte"):
    modo_tb = st.radio("Como obter o trecho-base?", ["Selecionar parágrafo(s)", "Resumo automático"], horizontal=True)
    if modo_tb == "Selecionar parágrafo(s)":
        paras = [p.strip() for p in st.session_state.texto_fonte.split("\n") if len(p.strip())>80]
        sel = st.multiselect(
            "Escolha parágrafo(s):", paras,
            format_func=lambda p: textwrap.shorten(p, 120, placeholder="…")
        )
        if sel:
            st.session_state.trecho = "\n\n".join(sel)
    else:
        if st.button("🔎 Resumir texto completo"):
            prompt = (
                "Resuma em até 3 frases este texto para servir de base "
                "a uma situação-problema ENADE:\n\n" + st.session_state.texto_fonte
            )
            resumo = gerar_llm(prompt, role_system="user", temperature=0.5, max_tokens=200)
            st.session_state.trecho = resumo
        if st.session_state.get("trecho"):
            st.session_state.trecho = st.text_area(
                "Resumo (edite se quiser)", st.session_state.trecho, height=150
            )

# --- 5. Contexto ---
st.header("4. Contexto (situação-problema)")
if st.session_state.get("trecho"):
    if not st.session_state.get("contexto"):
        prompt = (
            "Com base neste trecho, gere UMA BREVE situação-problema profissional "
            "e relevante para uma questão ENADE. Retorne apenas o texto:\n\n"
            + st.session_state.trecho
        )
        ctx = gerar_llm(prompt, role_system="user", temperature=0.7, max_tokens=300)
        st.session_state.contexto = ctx
    st.session_state.contexto = st.text_area(
        "Edite o contexto se necessário:", st.session_state.contexto, height=120
    )

# --- 6. Referência ABNT (prefilled & editável) ---
st.header("5. Referência ABNT")
if st.session_state.get("fonte_link"):
    hoje = datetime.now()
    meses = ["jan.","fev.","mar.","abr.","mai.","jun.","jul.","ago.","set.","out.","nov.","dez."]
    acesso = f"{hoje.day} {meses[hoje.month-1]} {hoje.year}"
    default_abnt = (
        f"{st.session_state.fonte_link}. Disponível em: <{st.session_state.fonte_link}>. "
        f"Acesso em: {acesso}."
    )
    st.session_state.referencia = st.text_area(
        "Referência ABNT (edite se quiser):", default_abnt, height=100
    )

# --- 7. Parâmetros ENADE & Bloom & Geração ---
st.header("6. Gerar Questão ENADE")
if all(k in st.session_state for k in ("contexto","trecho","referencia")):
    with st.form("enade_form"):
        tipo   = st.selectbox("Tipo de item", ["Múltipla Escolha", "Asserção-Razão", "Discursivo"])
        perfil = st.text_input("Perfil do egresso", placeholder="Ex: crítico e reflexivo")
        comp   = st.text_input("Competência", placeholder="Ex: analisar conflitos éticos")
        obj    = st.text_input("Objeto de conhecimento", placeholder="Ex: ética profissional")
        diff   = st.select_slider("Dificuldade", ["Fácil","Média","Difícil"], value="Média")
        extra  = st.text_area("Info. adicional (opcional)")

        st.subheader("Taxonomia de Bloom")
        modo_bloom = st.radio(
            "Como selecionar verbos de Bloom?",
            ["Por faixa de níveis", "Por nível único"], horizontal=True
        )
        if modo_bloom == "Por faixa de níveis":
            faixa = st.select_slider(
                "Faixa cognitiva (menor → maior):", options=BLOOM_LEVELS,
                value=(BLOOM_LEVELS[0], BLOOM_LEVELS[-1])
            )
            idx0, idx1 = BLOOM_LEVELS.index(faixa[0]), BLOOM_LEVELS.index(faixa[1])
            niveis = BLOOM_LEVELS[idx0:idx1+1]
            verbos = [v for lvl in niveis for v in BLOOM_VERBS[lvl]]
        else:
            nivel = st.selectbox("Nível cognitivo:", BLOOM_LEVELS)
            verbos = BLOOM_VERBS[nivel]
        selected_verbs = st.multiselect("Verbos de Bloom:", options=verbos)

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
- Área: {area}
- Curso: {curso}
- Assunto: {assunto}
- Tipo: {tipo}
- Perfil: {perfil}
- Competência: {comp}
- Objeto de conhecimento: {obj}
- Dificuldade: {diff}
- Verbos de Bloom: {', '.join(selected_verbs) if selected_verbs else 'nenhum especificado'}
- Observações: {extra}
"""
        raw = gerar_questao_llm(system_prompt, user_prompt)
        try:
            st.session_state.questao = json.loads(raw)
        except:
            st.session_state.questao = raw

# --- 8. Resultado & Gerar Outra Questão ---
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
    if st.button("🔄 Gerar outra questão"):
        for key in ("texto_fonte","trecho","contexto","referencia","questao"):
            st.session_state.pop(key, None)
        st.experimental_rerun()
