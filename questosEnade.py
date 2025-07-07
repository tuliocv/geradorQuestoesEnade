import streamlit as st
import os
import requests
import textwrap
import json
from datetime import datetime
from bs4 import BeautifulSoup
from openai import OpenAI
import PyPDF2
from io import BytesIO

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Gerador de Questões ENADE",
    page_icon="🎓",
    layout="wide"
)

# --- ESTADO DA SESSÃO ---
for key in ("texto_fonte", "trecho_para_prompt", "questao_bruta", "questao"):
    if key not in st.session_state:
        st.session_state[key] = "" if key != "questao" else None
if "last_pdf" not in st.session_state:
    st.session_state.last_pdf = None
if "fonte_info" not in st.session_state:
    st.session_state.fonte_info = {"link": ""}

# --- DICIONÁRIO DE ÁREAS ---
AREAS_ENADE = {
    "Ciências Sociais Aplicadas": [
        "Administração", "Arquitetura e Urbanismo", "Biblioteconomia",
        "Ciências Contábeis", "Ciências Econômicas", "Comunicação Social",
        "Direito", "Design", "Gestão de Políticas Públicas", "Jornalismo",
        "Publicidade e Propaganda", "Relações Internacionais", "Serviço Social",
        "Turismo"
    ],
    "Engenharias": [
        "Engenharia Aeronáutica", "Engenharia Agrícola", "Engenharia Ambiental",
        "Engenharia Biomédica", "Engenharia Cartográfica", "Engenharia Civil",
        "Engenharia de Alimentos", "Engenharia de Computação",
        "Engenharia de Controle e Automação", "Engenharia de Materiais",
        "Engenharia de Minas", "Engenharia de Petróleo", "Engenharia de Produção",
        "Engenharia de Software", "Engenharia Elétrica", "Engenharia Eletrônica",
        "Engenharia Florestal", "Engenharia Mecânica", "Engenharia Mecatrônica",
        "Engenharia Metalúrgica", "Engenharia Naval", "Engenharia Química",
        "Engenharia Têxtil"
    ],
    "Ciências da Saúde": [
        "Educação Física", "Enfermagem", "Farmácia", "Fisioterapia",
        "Fonoaudiologia", "Medicina", "Medicina Veterinária", "Nutrição",
        "Odontologia", "Saúde Coletiva"
    ],
}

# --- CHECKLIST E FORMATO DE SAÍDA ---
SYSTEM_PROMPT = """
Você é um docente especialista no ENADE (INEP). Siga rigorosamente este checklist:
1. Defina um "contexto" (situação-problema) breve e relevante.
2. Apresente um "texto_base" referenciado (Autor/Veículo, Ano, Link/Arquivo).
3. Elabore um "enunciado" afirmativo, claro e objetivo.
4. Gere exatamente 5 "alternativas" (A–E), apenas 1 correta.
5. Distratores plausíveis, baseados em erros comuns.
6. Use linguagem formal, impessoal, norma-padrão.
7. Avalie uma competência (aplicação de conhecimento), não memorização.
8. Evite termos absolutos (“sempre”, “nunca”, “somente”, etc.).
9. Indique "gabarito" no formato: "Letra X".
10. Inclua "justificativas" breves para cada alternativa.

**Formato de saída** (retorne apenas este JSON):
{
  "contexto": "...",
  "texto_base": "...",
  "referencia": "...",
  "enunciado": "...",
  "alternativas": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "...",
    "E": "..."
  },
  "gabarito": "Letra X",
  "justificativas": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "...",
    "E": "..."
  }
}
"""

# --- EXTRAÇÃO DE TEXTO ---
@st.cache_data(ttl=3600)
def extrair_texto_url(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        st.error(f"Falha ao extrair URL: {e}")
        return None

@st.cache_data
def extrair_texto_pdf(upload) -> str | None:
    try:
        reader = PyPDF2.PdfReader(BytesIO(upload.read()))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        st.error(f"Falha ao ler PDF: {e}")
        return None

# --- GERAÇÃO PELO OPENAI ---
def gerar_questao_llm(prompt: str, api_key: str, modelo: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return resp.choices[0].message.content

# --- SIDEBAR: CONFIGURAÇÃO DA API ---
with st.sidebar:
    st.markdown(
        "## 🔑 Configuração da API\n"
        "- **OpenAI**: platform.openai.com/account/api-keys"
    )
    api_key = st.text_input("Chave da OpenAI", type="password")
    modelo = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-3.5-turbo"])
    if not api_key:
        st.warning("Insira sua chave de API para continuar.")
        st.stop()

# --- ETAPA 1: ESCOPO ---
st.header("1. Definição do Escopo")
area    = st.selectbox("Grande Área", list(AREAS_ENADE.keys()))
curso   = st.selectbox("Curso", AREAS_ENADE[area])
assunto = st.text_input("Tópico/Assunto central")

# --- ETAPA 2: TEXTO-BASE ---
st.header("2. Texto-Base")
metodo = st.radio("Fonte do texto-base:", ["URL", "PDF"])
if metodo == "URL":
    url = st.text_input("Cole a URL:")
    if st.button("Extrair texto da URL"):
        txt = extrair_texto_url(url)
        if txt:
            st.session_state.texto_fonte = txt
            st.session_state.fonte_info["link"] = url
elif metodo == "PDF":
    up = st.file_uploader("Envie o PDF", type=["pdf"])
    if up and up != st.session_state.last_pdf:
        txt = extrair_texto_pdf(up)
        if txt:
            st.session_state.texto_fonte = txt
            st.session_state.fonte_info["link"] = up.name
            st.session_state.last_pdf = up

if st.session_state.texto_fonte:
    st.success("Texto-base carregado!")
    with st.expander("Ver texto extraído"):
        st.text_area("Texto-Fonte", st.session_state.texto_fonte, height=300)
    # seleção de parágrafos
    pars = [p for p in st.session_state.texto_fonte.split("\n") if len(p.strip()) > 100]
    sel = st.multiselect(
        "Selecione parágrafos para Texto-Base",
        options=pars,
        format_func=lambda x: textwrap.shorten(x, 100, placeholder="...")
    )
    st.session_state.trecho_para_prompt = "\n\n".join(sel) if sel else st.session_state.texto_fonte

# --- ETAPA 3: CONTEXTO E PARÂMETROS ---
if st.session_state.trecho_para_prompt:
    st.header("3. Contexto e Parâmetros ENADE")
    contexto = st.text_area("Contexto (situação-problema)", "")
    with st.form("enade_form"):
        fonte      = st.text_input("Fonte/Veículo", "")
        ano        = st.text_input("Ano", "")
        tipo_item  = st.selectbox("Tipo de item", ["Múltipla Escolha", "Asserção-Razão", "Discursivo"])
        perfil     = st.text_input("Perfil do egresso", "")
        competencia= st.text_input("Competência", "")
        objeto     = st.text_input("Objeto de conhecimento", "")
        dificuldade= st.select_slider("Dificuldade", ["Fácil", "Média", "Difícil"])
        info_add   = st.text_area("Observações (opcional)", "")
        submit     = st.form_submit_button("🚀 Gerar Questão")
    if submit:
        if not (fonte and ano and contexto):
            st.error("Preencha Fonte, Ano e Contexto.")
        else:
            # referência ABNT simplificado
            hoje = datetime.now()
            meses = ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.",
                     "jul.", "ago.", "set.", "out.", "nov.", "dez."]
            data_acesso = f"{hoje.day} {meses[hoje.month-1]} {hoje.year}"
            referencia = (
                f"Fonte: {fonte}, {ano}. Disponível em: "
                f"{st.session_state.fonte_info['link']}. Acesso em: {data_acesso}."
            )
            # montar prompt
            prompt = f"""
**Contexto (situação-problema):**
{contexto}

**Texto-Base:**
{st.session_state.trecho_para_prompt}

**Referência:**
{referencia}

**Encomenda ENADE:**
- Curso: {curso}
- Assunto: {assunto}
- Tipo de item: {tipo_item}
- Perfil do egresso: {perfil}
- Competência: {competencia}
- Objeto de conhecimento: {objeto}
- Dificuldade: {dificuldade}
- Observações: {info_add}
"""
            raw = gerar_questao_llm(prompt, api_key, modelo)
            st.session_state.questao_bruta = raw
            # validação JSON
            try:
                q = json.loads(raw)
                campos = {"contexto","texto_base","referencia","enunciado","alternativas","gabarito","justificativas"}
                faltando = campos - set(q.keys())
                if faltando:
                    st.error(f"Faltam campos na resposta: {faltando}")
                else:
                    st.session_state.questao = q
            except Exception as e:
                st.error(f"Resposta não é JSON válido: {e}")

# --- ETAPA 4: EXIBIÇÃO ---
if st.session_state.questao:
    st.header("4. Questão ENADE Estruturada")
    q = st.session_state.questao
    st.markdown(f"**Contexto:** {q['contexto']}")
    st.markdown(f"**Texto-Base:** {q['texto_base']}")
    st.markdown(f"**Referência:** {q['referencia']}")
    st.markdown(f"**Enunciado:** {q['enunciado']}")
    st.markdown("**Alternativas:**")
    for letra, texto in q["alternativas"].items():
        st.markdown(f"- **{letra}**: {texto}")
    st.markdown(f"**Gabarito:** {q['gabarito']}")
    st.markdown("**Justificativas:**")
    for letra, jus in q["justificativas"].items():
        st.markdown(f"- **{letra}**: {jus}")
    st.download_button(
        "📥 Baixar (.txt)",
        data=json.dumps(q, ensure_ascii=False, indent=2),
        file_name=f"questao_{curso.replace(' ','_')}.json",
        mime="application/json"
    )
else:
    st.info("Complete todas as etapas para gerar a questão.")
