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

# --- 1. Configuração da página & API ---
st.set_page_config(page_title="Gerador de Questões ENADE", page_icon="🎓", layout="wide")
st.sidebar.header("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("", type="password")
model   = st.sidebar.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-3.5-turbo"])
if not api_key:
    st.sidebar.warning("Informe sua chave da OpenAI para continuar.")
    st.stop()

# --- Funções auxiliares ---
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

# --- 2. Escopo da questão ---
st.header("1. Definição do Escopo")
area    = st.text_input("Área do conhecimento", placeholder="Ex: Engenharias")
curso   = st.text_input("Curso",               placeholder="Ex: Engenharia de Software")
assunto = st.text_input("Tópico/Assunto central", placeholder="Ex: IA na arbitragem")

# --- 3. Carregar texto-base ---
st.header("2. Carregue o Texto-Base")
metodo = st.radio("Origem do texto-base:", ["URL", "PDF"], horizontal=True)
if metodo == "URL":
    url = st.text_input("Cole a URL completa")
    if st.button("▶️ Extrair texto da URL"):
        txt = extrair_texto_url(url)
        if txt:
            st.session_state.full_text = txt
            st.session_state.link      = url
elif metodo == "PDF":
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

# --- 4. Texto-Base selecionado pela IA ou pelo usuário ---
st.header("3. Defina o Texto-Base")
if st.session_state.get("full_text"):
    modo_tb = st.radio("Como obter o trecho-base?", ["Selecionar manualmente", "Resumo automático"], horizontal=True)
    if modo_tb == "Selecionar manualmente":
        paras = [p.strip() for p in st.session_state.full_text.split("\n") if len(p.strip())>80]
        sel = st.multiselect(
            "Escolha parágrafo(s):", paras,
            format_func=lambda p: textwrap.shorten(p, 120, placeholder="…")
        )
        if sel:
            st.session_state.text_base = "\n\n".join(sel)
    else:
        if st.button("🔎 Gerar resumo automático"):
            prompt = [
                {"role":"system","content":"Você é um assistente que gera resumos concisos para questões ENADE."},
                {"role":"user","content":
                    "Resuma em até 3 frases este texto para servir de base a uma situação-problema ENADE:\n\n"
                    + st.session_state.full_text}
            ]
            st.session_state.text_base = chamar_llm(prompt, temperature=0.5, max_tokens=200)
    if st.session_state.get("text_base"):
        st.session_state.text_base = st.text_area(
            "Texto-Base (edite se desejar):",
            value=st.session_state.text_base, height=150
        )

# --- 5. Referência ABNT pré-preenchida e editável ---
st.header("4. Referência ABNT")
col1, col2, col3, col4 = st.columns(4)
autor_ref    = col1.text_input("Autor (SOBRENOME, Nome)", value=st.session_state.get("autor_ref",""))
titulo_ref   = col2.text_input("Título do texto-base", value=st.session_state.get("titulo_ref",""))
veiculo_ref  = col3.text_input("Veículo (site, jornal, revista)", value=st.session_state.get("veiculo_ref",""))
data_pub_ref = col4.text_input("Data de publicação (dd mmm. aaaa)", value=st.session_state.get("data_pub_ref",""))

st.session_state["autor_ref"]    = autor_ref
st.session_state["titulo_ref"]   = titulo_ref
st.session_state["veiculo_ref"]  = veiculo_ref
st.session_state["data_pub_ref"] = data_pub_ref

if autor_ref and titulo_ref and veiculo_ref and data_pub_ref and st.session_state.get("link"):
    hoje = datetime.now()
    meses_abnt = ["jan.","fev.","mar.","abr.","mai.","jun.","jul.","ago.","set.","out.","nov.","dez."]
    acesso = f"{hoje.day} {meses_abnt[hoje.month-1]} {hoje.year}"
    referencia_abnt = (
        f"{autor_ref}. {titulo_ref}. {veiculo_ref}, {data_pub_ref}. "
        f"Disponível em: <{st.session_state.link}>. Acesso em: {acesso}."
    )
    st.session_state["referencia"] = referencia_abnt
    st.session_state["referencia"] = st.text_area(
        "Referência ABNT (edite se quiser):", referencia_abnt, height=100
    )
else:
    st.info("Preencha Autor, Título, Veículo e Data para gerar referência ABNT.")

# --- 6. Parâmetros ENADE & Bloom ---
st.header("5. Parâmetros da Questão")
if "params_confirmed" not in st.session_state:
    st.session_state["params_confirmed"] = False

with st.form("params_form"):
    perfil      = st.text_input("Perfil do egresso")
    competencia = st.text_input("Competência")
    objeto      = st.text_input("Objeto de conhecimento")
    dificuldade = st.select_slider("Nível de dificuldade", ["Fácil","Média","Difícil"], value="Média")
    extra       = st.text_area("Observações adicionais (opcional)")

    st.subheader("Taxonomia de Bloom")
    modo_b = st.radio("Verbos de Bloom por:", ["Faixa de níveis","Nível único"], horizontal=True)
    if modo_b == "Faixa de níveis":
        faixa = st.select_slider("Faixa:", options=BLOOM_LEVELS,
                                 value=(BLOOM_LEVELS[0], BLOOM_LEVELS[-1]))
        i0, i1 = BLOOM_LEVELS.index(faixa[0]), BLOOM_LEVELS.index(faixa[1])
        verbs = [v for lvl in BLOOM_LEVELS[i0:i1+1] for v in BLOOM_VERBS[lvl]]
    else:
        lvl   = st.selectbox("Nível:", BLOOM_LEVELS)
        verbs = BLOOM_VERBS[lvl]
    selected_verbs = st.multiselect("Selecione verbos:", verbs)

    btn = st.form_submit_button("▶️ Confirmar Parâmetros")
    if btn:
        st.session_state["params_confirmed"] = True
        st.success("Parâmetros definidos.")

# --- 7. Geração de Contextualização ---
st.header("6. Contextualização (gerada pela IA)")
if (
    st.session_state.get("text_base")
    and st.session_state.params_confirmed
    and not st.session_state.get("context")
):
    messages = [
        {"role":"system","content":"Você elabora contextos para questões ENADE."},
        {"role":"user","content":
            f"Com base neste TEXTO-BASE e nos parâmetros:\n"
            f"- Perfil: {perfil}\n"
            f"- Competência: {competencia}\n"
            f"- Objeto: {objeto}\n"
            f"- Dificuldade: {dificuldade}\n"
            f"- Verbos de Bloom: {', '.join(selected_verbs)}\n"
            f"- Observações: {extra}\n\n"
            f"TEXTO-BASE:\n{st.session_state.text_base}\n\n"
            "Gere uma breve contextualização (situação-problema)."
        }
    ]
    st.session_state.context = chamar_llm(messages, temperature=0.7, max_tokens=200)

if st.session_state.get("context"):
    st.session_state.context = st.text_area(
        "Contextualização (edite se quiser):",
        value=st.session_state.context,
        height=120
    )

# --- 8. Geração da Questão ENADE ---
st.header("7. Gerar Questão ENADE")
if st.session_state.get("text_base") and st.session_state.get("context"):
    if st.button("🚀 Gerar Questão"):
        system_prompt = """
Você é docente especialista INEP. Crie uma questão padrão ENADE, seguindo rigorosamente:
- Texto-base e contextualização definidos acima
- Enunciado afirmativo, claro e objetivo
- 5 alternativas A–E, apenas 1 correta
- Distratores plausíveis baseados em erros comuns
- Linguagem formal, impessoal, norma-padrão
- Foco em aplicação de conhecimento (situação-problema)
- Evitar termos absolutos (sempre, nunca, apenas, etc.)
- Ao final, indique "Gabarito: Letra X"
- Inclua justificativas breves para cada alternativa
"""
        user_prompt = f"""
TEXTO-BASE:
{st.session_state.text_base}

CONTEXTUALIZAÇÃO:
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

Retorne em JSON:
{{
  "enunciado": "...",
  "alternativas":{{"A":"", "B":"", "C":"", "D":"", "E":""}},
  "gabarito": "Letra X",
  "justificativas":{{"A":"", "B":"", "C":"", "D":"", "E":""}}
}}
"""
        raw = chamar_llm(
            [{"role":"system","content":system_prompt},
             {"role":"user","content":user_prompt}],
            temperature=0.3, max_tokens=1000
        )
        try:
            st.session_state.questao = json.loads(raw)
        except:
            st.session_state.questao = raw

# --- 9. Exibição e download em Word ---
st.header("8. Resultado")
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

    # Gerar Word
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

    if st.button("🔄 Gerar outra questão"):
        for k in ("full_text","text_base","autor_ref","titulo_ref","veiculo_ref",
                  "data_pub_ref","referencia","params_confirmed","context","questao"):
            st.session_state.pop(k, None)
        st.experimental_rerun()
