import streamlit as st
import os
import requests
import textwrap
import pandas as pd
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

# Estado da sessão
if 'texto_fonte' not in st.session_state:
    st.session_state.texto_fonte = ""
if 'fonte_info' not in st.session_state:
    st.session_state.fonte_info = {"source": "", "year": "", "link": ""}
if 'trecho_para_prompt' not in st.session_state:
    st.session_state.trecho_para_prompt = ""
if 'usar_contextualizacao_ia' not in st.session_state:
    st.session_state.usar_contextualizacao_ia = False
if 'questao_gerada' not in st.session_state:
    st.session_state.questao_gerada = ""

# --- DICIONÁRIO DE ÁREAS ---

AREAS_ENADE = {
    "Ciências Sociais Aplicadas": [
        "Administração", "Arquitetura e Urbanismo", "Biblioteconomia", "Ciências Contábeis",
        "Ciências Econômicas", "Comunicação Social", "Direito", "Design", "Gestão de Políticas Públicas",
        "Jornalismo", "Publicidade e Propaganda", "Relações Internacionais", "Serviço Social",
        "Turismo"
    ],
    "Engenharias": [
        "Engenharia Aeronáutica", "Engenharia Agrícola", "Engenharia Ambiental", "Engenharia Biomédica",
        "Engenharia Cartográfica", "Engenharia Civil", "Engenharia de Alimentos", "Engenharia de Computação",
        "Engenharia de Controle e Automação", "Engenharia de Materiais", "Engenharia de Minas",
        "Engenharia de Petróleo", "Engenharia de Produção", "Engenharia de Software", "Engenharia Elétrica",
        "Engenharia Eletrônica", "Engenharia Florestal", "Engenharia Mecânica", "Engenharia Mecatrônica",
        "Engenharia Metalúrgica", "Engenharia Naval", "Engenharia Química", "Engenharia Têxtil"
    ],
    "Ciências da Saúde": [
        "Educação Física", "Enfermagem", "Farmácia", "Fisioterapia", "Fonoaudiologia",
        "Medicina", "Medicina Veterinária", "Nutrição", "Odontologia", "Saúde Coletiva"
    ],
}

# --- REQUISITOS OBRIGATÓRIOS DO ENADE ---

REQUISITOS_OBRIGATORIOS_ENADE = """
- **Originalidade e Ineditismo**: A questão deve ser totalmente inédita.
- **Estrutura do Item**: Deve conter um texto-base (situação-estímulo), um enunciado claro e 5 alternativas (A, B, C, D, E).
- **Texto-Base**: Deve ser indispensável para a resolução da questão, não apenas um pretexto. A fonte completa (Autor/Veículo, Ano, Link/Nome do Arquivo) é obrigatória.
- **Enunciado**: Deve ser uma instrução clara, objetiva e formulada de maneira afirmativa. Não deve solicitar a "incorreta" ou a "exceção".
- **Alternativa Correta (Gabarito)**: Apenas UMA alternativa deve ser inquestionavelmente correta.
- **Distratores**: As quatro alternativas incorretas (distratores) devem ser plausíveis, baseadas em erros comuns ou interpretações equivocadas, mas claramente erradas para quem domina o conteúdo.
- **Linguagem**: A linguagem deve ser formal, impessoal, precisa e seguir a norma-padrão.
- **Foco em Competências**: A questão deve avaliar a aplicação do conhecimento para resolver uma situação-problema, não a simples memorização de conceitos.
- **Evitar Termos Problemáticos**: Evitar o uso de termos como "sempre", "nunca", "todos", "nenhum", "apenas", "somente" nas alternativas.
"""

# --- FUNÇÕES AUXILIARES ---

@st.cache_data(ttl=3600)
def extrair_texto_url(url: str) -> str | None:
    """Extrai texto de uma página web usando requests + BeautifulSoup."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        st.error(f"Falha ao extrair texto da URL: {e}")
        return None

@st.cache_data
def extrair_texto_pdf(arquivo_pdf) -> str | None:
    """Extrai texto de um arquivo PDF carregado."""
    try:
        leitor = PyPDF2.PdfReader(BytesIO(arquivo_pdf.read()))
        texto = ""
        for pagina in leitor.pages:
            texto += pagina.extract_text() or ""
        return texto
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return None

def gerar_questao_com_llm(prompt: str, provedor: str, api_key: str, modelo: str) -> str | None:
    """Gera a questão chamando a API do modelo de IA escolhido."""
    try:
        if provedor == "ChatGPT (OpenAI)":
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": f"Você é um docente especialista do INEP e deve criar uma questão para o ENADE. Siga RIGOROSAMENTE as regras:\n{REQUISITOS_OBRIGATORIOS_ENADE}"},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1500
            )
            return resp.choices[0].message.content

        elif provedor == "Gemini (Google)":
            genai.configure(api_key=api_key)
            gm = genai.GenerativeModel(modelo)
            full_prompt = f"Como docente especialista do INEP, crie uma questão ENADE seguindo estas regras:\n{REQUISITOS_OBRIGATORIOS_ENADE}\n\nEncomenda:\n{prompt}"
            resp = gm.generate_content(full_prompt)
            return resp.text

    except Exception as e:
        st.error(f"Erro ao chamar a API de {provedor}: {e}")
        return None

# --- INTERFACE DO STREAMLIT ---

st.title("🎓 Assistente para Elaboração de Questões ENADE")
st.markdown("Este app cria questões ENADE seguindo as diretrizes oficiais do INEP em 4 etapas.")

# --- SIDEBAR: CONFIGURAÇÃO DE IA ---
with st.sidebar:
    st.markdown(
        "## 🔑 Configuração da IA\n"
        "**Como obter sua chave de API**\n\n"
        "- **OpenAI**: platform.openai.com/account/api-keys\n"
        "- **Google Gemini**: Console Google Cloud → Generative AI → Chaves de API\n"
    )
    provedor_ia = st.selectbox("Provedor de IA", ["ChatGPT (OpenAI)", "Gemini (Google)"])
    default_key = (
        st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if provedor_ia == "ChatGPT (OpenAI)"
        else st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    api_key = st.text_input("Chave de API", value=default_key or "", type="password")
    modelo_selecionado = st.selectbox(
        "Modelo",
        ["gpt-4o", "gpt-3.5-turbo"] if provedor_ia.startswith("ChatGPT")
        else ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
    )
    if not api_key:
        st.warning("Insira sua chave de API para continuar.")
        st.stop()

# --- ETAPA 1: ESCOPO ---
st.header("Etapa 1: Definição do Escopo")
col1, col2 = st.columns(2)
with col1:
    area = st.selectbox("Grande Área do Conhecimento", list(AREAS_ENADE.keys()))
with col2:
    curso = st.selectbox("Curso", AREAS_ENADE[area])
assunto = st.text_input("Assunto ou tópico central", placeholder="Ex: Estruturas de Controle em algoritmos")

# --- ETAPA 2: TEXTO-BASE ---
st.header("Etapa 2: Texto-Base (Situação-Estímulo)")
tab_url, tab_pdf = st.tabs(["🔗 URL", "📄 PDF"])

with tab_url:
    url_artigo = st.text_input("URL do artigo/página:")
    if st.button("Extrair da URL"):
        st.session_state.texto_fonte = extrair_texto_url(url_artigo)
        st.session_state.fonte_info.update({"link": url_artigo})

with tab_pdf:
    pdf_file = st.file_uploader("Faça upload do PDF", type=["pdf"])
    if pdf_file:
        st.session_state.texto_fonte = extrair_texto_pdf(pdf_file)
        st.session_state.fonte_info.update({"link": pdf_file.name})

# --- ETAPA 3: PREPARAÇÃO DA ENCOMENDA ---
st.header("Etapa 3: Preparação da Encomenda")
if st.session_state.texto_fonte:
    st.success("Material de base pronto!")
    with st.expander("Ver texto extraído"):
        st.text_area("Texto-Fonte", st.session_state.texto_fonte, height=300)

    modo = st.radio(
        "Uso do texto-fonte:",
        ["Usar parágrafos selecionados", "Gerar novo Texto-Base pela IA"],
        key="modo"
    )
    st.session_state.usar_contextualizacao_ia = (modo == "Gerar novo Texto-Base pela IA")

    if not st.session_state.usar_contextualizacao_ia:
        pars = [p for p in st.session_state.texto_fonte.split("\n") if len(p.strip()) > 100]
        selecionados = st.multiselect(
            "Selecione parágrafos para texto-base",
            options=pars,
            format_func=lambda p: textwrap.shorten(p, 100, placeholder="...")
        )
        if selecionados:
            st.session_state.trecho_para_prompt = "\n\n".join(selecionados)
        else:
            st.warning("Nenhum parágrafo longo encontrado; usará todo o texto.")
            st.session_state.trecho_para_prompt = st.session_state.texto_fonte
    else:
        st.info("A IA criará um novo Texto-Base a partir de todo o documento.")
        st.session_state.trecho_para_prompt = st.session_state.texto_fonte

    with st.form("encomenda"):
        fonte = st.text_input("Fonte/Veículo", placeholder="Ex: G1, Livro X")
        ano = st.text_input("Ano de Publicação", placeholder="Ex: 2024")
        tipo_item = st.selectbox("Tipo de item", ["Múltipla Escolha", "Asserção-Razão", "Discursivo"])
        perfil = st.text_input("Perfil do egresso", placeholder="Ex: Ético e reflexivo")
        competencia = st.text_input("Competência", placeholder="Ex: Analisar conflitos éticos")
        objeto = st.text_input("Objeto de conhecimento", placeholder="Ex: Legislação e ética")
        dificuldade = st.select_slider("Dificuldade", ["Fácil", "Média", "Difícil"], value="Média")
        info_add = st.text_area("Instrução adicional (opcional)")

        if st.form_submit_button("🚀 Gerar Questão"):
            if not fonte or not ano or not st.session_state.trecho_para_prompt:
                st.error("Preencha Fonte, Ano e selecione o texto-base.")
            else:
                hoje = datetime.now()
                meses = ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]
                data_acesso = f"{hoje.day} {meses[hoje.month-1]} {hoje.year}"
                fonte_str = (
                    f"Fonte: {fonte}, {ano}. Disponível em: "
                    f"<{st.session_state.fonte_info['link']}>. Acesso em: {data_acesso}."
                )

                if st.session_state.usar_contextualizacao_ia:
                    instrucao_tb = (
                        "**1. CRIAR NOVO TEXTO-BASE:**\n"
                        f"{st.session_state.trecho_para_prompt}\n\n"
                        "Em seguida, elabore a questão completa."
                    )
                else:
                    instrucao_tb = (
                        "**1. TEXTO-BASE LITERAL:**\n"
                        f"{st.session_state.trecho_para_prompt}"
                    )

                prompt_final = f"""
**ENCOMENDA ENADE**

{instrucao_tb}

{fonte_str}

**Dados da Encomenda:**
- Curso: {curso}
- Assunto: {assunto}
- Tipo de item: {tipo_item}
- Perfil do egresso: {perfil}
- Competência: {competencia}
- Objeto de conhecimento: {objeto}
- Dificuldade: {dificuldade}
- Instrução adicional: {info_add}

**Tarefa:** Gere a questão completa com:
1) Texto-base (ABNT simplificado);
2) Enunciado claro;
3) Cinco alternativas (A-E);
4) Gabarito no final: "Gabarito: Letra X".
"""
                st.session_state.questao_gerada = gerar_questao_com_llm(
                    prompt_final, provedor_ia, api_key, modelo_selecionado
                )

# --- ETAPA 4: RESULTADO ---
st.header("Etapa 4: Questão Gerada")
if st.session_state.questao_gerada:
    st.markdown(st.session_state.questao_gerada)
    st.download_button(
        "📥 Baixar como .txt",
        data=st.session_state.questao_gerada,
        file_name=f"questao_{curso.replace(' ', '_')}.txt",
        mime="text/plain"
    )
else:
    st.info("Complete as etapas acima para gerar sua questão ENADE.")
