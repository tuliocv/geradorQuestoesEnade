import streamlit as st
import os
import requests
import textwrap
from datetime import datetime
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
import PyPDF2
from io import BytesIO

# --- CONFIGURAÇÃO DA PÁGINA E ESTADO DA SESSÃO ---
st.set_page_config(
    page_title="Gerador de Questões ENADE",
    page_icon="🎓",
    layout="wide"
)

# Inicializa estados de sessão
for key in ("texto_fonte", "trecho_para_prompt", "questao_gerada"):
    if key not in st.session_state:
        st.session_state[key] = ""
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

# --- REGRAS OBRIGATÓRIAS DO ENADE ---
REQUISITOS_ENADE = """
- Originalidade total (sem reprises de provas antigas).
- Texto-base imprescindível; referenciar Autor/Veículo, Ano, Link/Arquivo.
- Enunciado afirmativo, claro e objetivo.
- 5 alternativas (A–E), apenas 1 correta.
- Distratores plausíveis, mas incorretos.
- Linguagem formal, impessoal, norma-padrão.
- Foco em resolver situação-problema (não memorização).
- Evitar “sempre”, “nunca”, “todos”, “nenhum”, “apenas”, “somente”.
"""

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600)
def extrair_texto_url(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
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

def gerar_questao(prompt: str, provedor: str, api_key: str, modelo: str) -> str | None:
    try:
        if provedor == "ChatGPT (OpenAI)":
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": f"Você é docente especialista ENADE. Siga estas regras:\n{REQUISITOS_ENADE}"},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1500
            )
            return r.choices[0].message.content

        else:  # Gemini
            genai.configure(api_key=api_key)
            gm = genai.GenerativeModel(modelo)
            full = f"Como especialista ENADE, siga estas regras:\n{REQUISITOS_ENADE}\n\n{prompt}"
            resp = gm.generate_content(full)
            return resp.text

    except Exception as e:
        st.error(f"Erro ao chamar API ({provedor}): {e}")
        return None

# --- SIDEBAR: CONFIGURAÇÃO DA API ---
with st.sidebar:
    st.markdown(
        "## 🔑 Configuração da API\n"
        "- **OpenAI GPT**: platform.openai.com/account/api-keys\n"
        "- **Google Gemini**: Google Cloud Console → Generative AI → API Keys\n"
    )
    provedor = st.selectbox("Provedor de IA", ["ChatGPT (OpenAI)", "Gemini (Google)"])
    default = (
        st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if provedor.startswith("ChatGPT")
        else st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    api_key = st.text_input("Chave de API", default or "", type="password")
    modelo = st.selectbox(
        "Modelo",
        ["gpt-4o-mini", "gpt-3.5-turbo"] if provedor.startswith("ChatGPT")
        else ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
    )
    if not api_key:
        st.warning("Insira sua chave de API para continuar.")
        st.stop()

# --- ETAPA 1: ESCOPO ---
st.header("1. Definição do Escopo")
area  = st.selectbox("Grande Área", list(AREAS_ENADE.keys()))
curso = st.selectbox("Curso", AREAS_ENADE[area])
assunto = st.text_input("Tópico/Assunto central", "")

# --- ETAPA 2: TEXTO-BASE ---
st.header("2. Texto-Base (Situação-Estímulo)")
metodo = st.radio("Fonte do texto-base:", ["Copiar link (URL)", "Fazer upload de PDF"])
if metodo == "Copiar link (URL)":
    url = st.text_input("Cole a URL aqui:")
    if st.button("Extrair da URL"):
        st.session_state.texto_fonte = extrair_texto_url(url)
        st.session_state.fonte_info["link"] = url
        st.experimental_rerun()
else:
    upload_pdf = st.file_uploader("Envie o PDF", type=["pdf"])
    if upload_pdf:
        st.session_state.texto_fonte = extrair_texto_pdf(upload_pdf)
        st.session_state.fonte_info["link"] = upload_pdf.name
        st.experimental_rerun()

if st.session_state.texto_fonte:
    st.success("Texto-base carregado!")
    with st.expander("Ver texto extraído"):
        st.text_area("Texto-Fonte", st.session_state.texto_fonte, height=300)

    modo = st.radio("Uso do texto-base:", ["Selecionar parágrafos", "Gerar novo pela IA"])
    use_ia = modo == "Gerar novo pela IA"
    st.session_state.usar_contextualizacao_ia = use_ia

    if not use_ia:
        paras = [
            p for p in st.session_state.texto_fonte.split("\n")
            if len(p.strip()) > 100
        ]
        sel = st.multiselect(
            "Selecione parágrafos para texto-base",
            options=paras,
            format_func=lambda x: textwrap.shorten(x, 100, placeholder="...")
        )
        if sel:
            st.session_state.trecho_para_prompt = "\n\n".join(sel)
        else:
            st.warning("Nenhum parágrafo longo encontrado; usará todo o texto.")
            st.session_state.trecho_para_prompt = st.session_state.texto_fonte
    else:
        st.info("A IA criará um novo texto-base a partir do documento inteiro.")
        st.session_state.trecho_para_prompt = st.session_state.texto_fonte

# --- ETAPA 3: PARÂMETROS DA ENCOMENDA ---
st.header("3. Parâmetros ENADE")
if st.session_state.trecho_para_prompt:
    with st.form("enade_form"):
        fonte   = st.text_input("Fonte/Veículo", "")
        ano     = st.text_input("Ano de publicação", "")
        tipo    = st.selectbox("Tipo de item", ["Múltipla Escolha", "Asserção-Razão", "Discursivo"])
        perfil  = st.text_input("Perfil do egresso", "")
        comp    = st.text_input("Competência", "")
        obj     = st.text_input("Objeto de conhecimento", "")
        diff    = st.select_slider("Dificuldade", ["Fácil", "Média", "Difícil"], value="Média")
        extra   = st.text_area("Observações adicionais (opcional)", "")
        submit = st.form_submit_button("🚀 Gerar Questão")

        if submit:
            if not (fonte and ano):
                st.error("Por favor, preencha 'Fonte/Veículo' e 'Ano de publicação'.")
            else:
                hoje = datetime.now()
                meses = ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.",
                         "jul.", "ago.", "set.", "out.", "nov.", "dez."]
                acesso = f"{hoje.day} {meses[hoje.month-1]} {hoje.year}"
                ref = (
                    f"Fonte: {fonte}, {ano}. Disponível em: "
                    f"{st.session_state.fonte_info['link']}. Acesso em: {acesso}."
                )

                if use_ia:
                    instr = (
                        "**1. CRIAR NOVO TEXTO-BASE:**\n"
                        f"{st.session_state.trecho_para_prompt}\n\n"
                        "Em seguida, elabore a questão completa."
                    )
                else:
                    instr = (
                        "**1. TEXTO-BASE LITERAL:**\n"
                        f"{st.session_state.trecho_para_prompt}"
                    )

                prompt = f"""
**ENCOMENDA ENADE**

{instr}

{ref}

- Curso: {curso}
- Assunto: {assunto}
- Tipo de item: {tipo}
- Perfil do egresso: {perfil}
- Competência: {comp}
- Objeto de conhecimento: {obj}
- Dificuldade: {diff}
- Observações: {extra}

**Tarefa:** Gere a questão completa contendo:
1) Texto-base (ABNT simplificado);
2) Enunciado claro e objetivo;
3) Cinco alternativas (A–E);
4) Gabarito: "Gabarito: Letra X".
"""
                st.session_state.questao_gerada = gerar_questao(
                    prompt, provedor, api_key, modelo
                )

# --- ETAPA 4: RESULTADO FINAL ---
st.header("4. Questão Gerada")
if st.session_state.questao_gerada:
    st.markdown(st.session_state.questao_gerada)
    st.download_button(
        "📥 Baixar (.txt)",
        data=st.session_state.questao_gerada,
        file_name=f"questao_{curso.replace(' ', '_')}.txt",
        mime="text/plain"
    )
else:
    st.info("Preencha todas as etapas para gerar sua questão ENADE.")
