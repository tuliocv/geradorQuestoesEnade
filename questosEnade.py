import streamlit as st
import requests
import textwrap
import json
from datetime import datetime
from io import BytesIO
from bs4 import BeautifulSoup
from openai import OpenAI
import PyPDF2
from docx import Document

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
model   = st.sidebar.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-3.5-turbo"])
if not api_key:
    st.sidebar.warning("Informe sua chave da OpenAI para continuar.")
    st.stop()

# --- AUXILIARES ---
@st.cache_data(ttl=3600)
def extrair_texto_url(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        st.error(f"Erro ao extrair URL: {e}")
        return None

@st.cache_data
def extrair_texto_pdf(upload) -> str | None:
    try:
        reader = PyPDF2.PdfReader(BytesIO(upload.read()))
        return "".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        st.error(f"Erro ao ler PDF: {e}")
        return None

def chamar_llm(messages, temperature=0.7, max_tokens=300):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- 2. ESCOPO ---
st.header("1. Definição do Escopo")
area    = st.text_input("Área do conhecimento", placeholder="Ex: Engenharias")
curso   = st.text_input("Curso",              placeholder="Ex: Engenharia de Software")
assunto = st.text_input("Tópico / Assunto central", placeholder="Ex: IA na arbitragem")

# --- 3. CARREGAR TEXTO-BASE ---
st.header("2. Carregue o Texto-Base")
metodo = st.radio("Origem:", ["URL","PDF"], horizontal=True)
if metodo=="URL":
    url = st.text_input("Cole a URL completa")
    if st.button("▶️ Extrair de URL"):
        txt = extrair_texto_url(url)
        if txt:
            st.session_state.full_text = txt
            st.session_state.link      = url
elif metodo=="PDF":
    pdf = st.file_uploader("Envie um PDF", type="pdf")
    if pdf:
        txt = extrair_texto_pdf(pdf)
        if txt:
            st.session_state.full_text = txt
            st.session_state.link      = pdf.name

if st.session_state.get("full_text"):
    with st.expander("Ver / editar texto-base completo"):
        st.session_state.full_text = st.text_area(
            "Texto completo", st.session_state.full_text, height=300
        )

# --- 4. GERAR TEXTO-BASE pelo LLM ---
st.header("3. Texto-Base (selecionado pela IA)")
if st.session_state.get("full_text") and not st.session_state.get("text_base"):
    messages = [
        {"role":"system","content":"Você é um assistente que seleciona trechos para questões ENADE."},
        {"role":"user","content":
            "Escolha um trecho de 100–200 palavras deste texto para servir de TEXTO-BASE em uma questão ENADE:\n\n"
            + st.session_state.full_text}
    ]
    tb = chamar_llm(messages, temperature=0.5, max_tokens=200)
    st.session_state.text_base = tb

if st.session_state.get("text_base"):
    st.session_state.text_base = st.text_area(
        "Trecho selecionado (edite se desejar):",
        value=st.session_state.text_base,
        height=150
    )

# --- 5. PARÂMETROS & BLOOM ---
st.header("4. Parâmetros da Questão")
with st.form("params_form"):
    perfil      = st.text_input("Perfil do egresso")
    competencia = st.text_input("Competência")
    objeto      = st.text_input("Objeto de conhecimento")
    dificuldade = st.select_slider("Nível de dificuldade", ["Fácil","Média","Difícil"], value="Média")
    extra       = st.text_area("Observações adicionais (opcional)")

    st.subheader("Taxonomia de Bloom")
    modo_b     = st.radio("Verbos de Bloom por:", ["Faixa de níveis","Nível único"], horizontal=True)
    if modo_b=="Faixa de níveis":
        faixa = st.select_slider("Faixa:", options=BLOOM_LEVELS, value=(BLOOM_LEVELS[0], BLOOM_LEVELS[-1]))
        i0,i1 = BLOOM_LEVELS.index(faixa[0]), BLOOM_LEVELS.index(faixa[1])
        verbs = [v for lvl in BLOOM_LEVELS[i0:i1+1] for v in BLOOM_VERBS[lvl]]
    else:
        lvl   = st.selectbox("Nível:", BLOOM_LEVELS)
        verbs = BLOOM_VERBS[lvl]
    selected_verbs = st.multiselect("Selecione verbos:", verbs)

    gerar_params = st.form_submit_button("▶️ Confirmar Parâmetros")

if gerar_params:
    st.success("Parâmetros definidos.")

# --- 6. GERAR CONTEXTUALIZAÇÃO pelo LLM ---
st.header("5. Contextualização (gerada pela IA)")
if st.session_state.get("text_base") and gerar_params and not st.session_state.get("context"):
    messages = [
        {"role":"system","content":"Você elabora contextos para questões ENADE."},
        {"role":"user","content":
            f"Com base neste TEXTO-BASE e nos parâmetros:\n"
            f"Perfil: {perfil}\nCompetência: {competencia}\nObjeto: {objeto}\n"
            f"Dificuldade: {dificuldade}\nVerbos de Bloom: {', '.join(selected_verbs)}\n"
            f"Observações: {extra}\n\n"
            f"TEXTO-BASE:\n{st.session_state.text_base}\n\n"
            "Gere uma breve contextualização (situação-problema)."
        }
    ]
    ctx = chamar_llm(messages, temperature=0.7, max_tokens=200)
    st.session_state.context = ctx

if st.session_state.get("context"):
    st.session_state.context = st.text_area(
        "Contextualização (edite se quiser):",
        value=st.session_state.context,
        height=120
    )

# --- 7. GERAR QUESTÃO FINAL ---
st.header("6. Gerar Questão ENADE")
if st.session_state.get("text_base") and st.session_state.get("context"):
    if st.button("🚀 Gerar Questão"):
        system_prompt = """
Você é docente especialista INEP. Crie uma questão padrão ENADE, seguindo rigorosamente:
- Texto-base definido acima
- Contextualização definida acima
- Enunciado afirmativo, claro e objetivo
- 5 alternativas (A–E), só 1 correta
- Distratores plausíveis
- Linguagem formal, impessoal, norma-padrão
- Foco em aplicação (situação-problema)
- Evitar termos absolutos (sempre,nunca,apenas,etc.)
- Ao final, indique "Gabarito: Letra X"
- Inclua justificativas breves para cada alternativa
"""
        user_prompt = f"""
TEXTO-BASE:
{st.session_state.text_base}

CONTEXTO:
{st.session_state.context}

Parâmetros:
- Área: {area}
- Curso: {curso}
- Assunto: {assunto}
- Perfil: {perfil}
- Competência: {competencia}
- Objeto de conhecimento: {objeto}
- Dificuldade: {dificuldade}
- Verbos de Bloom: {', '.join(selected_verbs) if selected_verbs else 'nenhum'}
- Observações: {extra}

Agora, gere o ENUNCIADO da questão, as 5 alternativas (A–E), o gabarito e as justificativas em JSON:
{{
  "enunciado": "...",
  "alternativas":{{"A":"", "B":"", "C":"", "D":"", "E":""}},
  "gabarito": "Letra X",
  "justificativas":{{"A":"", "B":"", "C":"", "D":"", "E":""}}
}}
"""
        raw = chamar_llm([{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}],
                         temperature=0.3, max_tokens=1000)
        try:
            st.session_state.questao = json.loads(raw)
        except:
            st.session_state.questao = raw

# --- 8. RESULTADO & DOWNLOAD EM WORD ---
st.header("7. Resultado")
q = st.session_state.get("questao")
if q:
    # Exibir em tela
    if isinstance(q, dict):
        st.subheader("Enunciado")
        st.markdown(q["enunciado"])
        st.subheader("Alternativas")
        for letra, texto in q["alternativas"].items():
            st.markdown(f"- **{letra}.** {texto}")
        st.markdown(f"**Gabarito:** {q['gabarito']}")
        st.subheader("Justificativas")
        for letra, jus in q["justificativas"].items():
            st.markdown(f"- **{letra}.** {jus}")
    else:
        st.markdown(q)

    # Gerar e baixar documento Word
    doc = Document()
    doc.add_heading("Questão ENADE", level=1)
    doc.add_paragraph("Texto-Base:")
    doc.add_paragraph(st.session_state.text_base)
    doc.add_paragraph("Contextualização:")
    doc.add_paragraph(st.session_state.context)
    if isinstance(q, dict):
        doc.add_paragraph("Enunciado:")
        doc.add_paragraph(q["enunciado"])
        doc.add_paragraph("Alternativas:")
        for letra, texto in q["alternativas"].items():
            doc.add_paragraph(f"{letra}. {texto}")
        doc.add_paragraph(f"Gabarito: {q['gabarito']}")
        doc.add_paragraph("Justificativas:")
        for letra, jus in q["justificativas"].items():
            doc.add_paragraph(f"{letra}. {jus}")

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    st.download_button(
        "📥 Baixar em Word (.docx)",
        data=buffer,
        file_name="questao_enade.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
