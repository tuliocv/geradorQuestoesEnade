import streamlit as st
import os
import requests
import textwrap
import pandas as pd
from datetime import datetime
from newspaper import Article
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

# Inicializar o estado da sessão para armazenar dados
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
def extrair_texto_url(url):
    """Extrai o texto principal de um artigo online."""
    try:
        art = Article(url, language='pt')
        art.download()
        art.parse()
        return art.text
    except Exception as e:
        st.error(f"Não foi possível extrair o conteúdo do artigo. Tente outro. Erro: {e}")
        return None

@st.cache_data
def extrair_texto_pdf(arquivo_pdf):
    """Extrai texto de um arquivo PDF carregado."""
    try:
        leitor_pdf = PyPDF2.PdfReader(BytesIO(arquivo_pdf.read()))
        texto_completo = ""
        for pagina in leitor_pdf.pages:
            texto_completo += pagina.extract_text()
        return texto_completo
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return None

def gerar_questao_com_llm(prompt, modelo, api_key):
    """Gera a questão chamando a API do modelo de IA escolhido."""
    try:
        if modelo == "ChatGPT (OpenAI)":
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Você é um docente especialista do INEP e deve criar uma questão para o ENADE. Siga RIGOROSAMENTE as seguintes regras oficiais: {REQUISITOS_OBRIGATORIOS_ENADE}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1500
            )
            return completion.choices[0].message.content

        elif modelo == "Gemini (Google)":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('Gemini 1.5 Flash')
            full_prompt = f"Como um docente especialista do INEP, sua tarefa é criar uma questão para o ENADE. Siga obrigatoriamente as regras abaixo:\n\n{REQUISITOS_OBRIGATORIOS_ENADE}\n\nAgora, com base na encomenda a seguir, gere a questão:\n\n{prompt}"
            response = model.generate_content(full_prompt)
            return response.text

    except Exception as e:
        st.error(f"Erro na comunicação com a API de {modelo}: {e}")
        return None


# --- INTERFACE DO STREAMLIT ---

st.title("🎓 Assistente para Elaboração de Questões ENADE")
st.markdown("Este aplicativo auxilia na criação de questões para o ENADE, seguindo as diretrizes oficiais do INEP. O processo é dividido em 4 etapas.")

# --- BARRA LATERAL PARA CONFIGURAÇÕES ---
with st.sidebar:
    st.header("🔑 Configuração da IA")
    modelo_ia = st.selectbox("Escolha o modelo de IA", ["ChatGPT (OpenAI)", "Gemini (Google)"])

    api_key = ""
    if modelo_ia == "ChatGPT (OpenAI)":
        api_key = st.text_input("Sua Chave de API da OpenAI", type="password", help="Obrigatorio para usar o ChatGPT.")
    elif modelo_ia == "Gemini (Google)":
        api_key = st.text_input("Sua Chave de API do Google AI", type="password", help="Obrigatorio para usar o Gemini.")

if not api_key:
    st.warning("Por favor, insira a chave de API na barra lateral para continuar.")
    st.stop()


# --- ETAPA 1: DEFINIÇÃO DO ESCOPO ---
st.header("Etapa 1: Definição do Escopo da Questão")

col1, col2 = st.columns(2)
with col1:
    area_escolhida = st.selectbox("Selecione a Grande Área do Conhecimento", list(AREAS_ENADE.keys()))
with col2:
    curso_escolhido = st.selectbox("Selecione o Curso", AREAS_ENADE[area_escolhida])

assunto = st.text_input("Qual o assunto ou tópico central da questão?", placeholder="Ex: Uso da IA na atribuição de processos de arbitragem")


# --- ETAPA 2: FORNECIMENTO DO TEXTO-BASE ---
st.header("Etapa 2: Fornecimento do Texto-Base (Situação-Estímulo)")

tab_url, tab_pdf = st.tabs(["🔗 Fornecer URL de Artigo", "📄 Carregar Arquivo PDF"])

with tab_url:
    url_artigo = st.text_input("Insira a URL do artigo ou página da web:")
    if st.button("Analisar URL"):
        with st.spinner("Analisando e extraindo texto da URL..."):
            st.session_state.texto_fonte = extrair_texto_url(url_artigo)
            st.session_state.fonte_info['link'] = url_artigo

with tab_pdf:
    arquivo_pdf = st.file_uploader("Carregue um arquivo PDF:", type=['pdf'])
    if arquivo_pdf is not None:
        with st.spinner("Analisando e extraindo texto do PDF..."):
            st.session_state.texto_fonte = extrair_texto_pdf(arquivo_pdf)
            st.session_state.fonte_info['link'] = arquivo_pdf.name


# --- ETAPA 3: DEFINIÇÃO DO USO DO TEXTO-BASE E ENCOMENDA ---
st.header("Etapa 3: Preparação do Item e Encomenda")

if st.session_state.texto_fonte:
    st.success("Material de base carregado com sucesso!")
    with st.expander("Ver o texto extraído"):
        st.text_area("Texto Completo", st.session_state.texto_fonte, height=300)

    st.subheader("3.1. Como usar o material de base?")
    modo_uso = st.radio(
        "Escolha como o texto fornecido será utilizado:",
        [
            "Selecionar um ou mais parágrafos para usar como texto-base literal.",
            "Pedir para a IA criar um novo texto-base conciso e contextualizado a partir do documento inteiro."
        ],
        key="modo_uso_radio"
    )
    st.session_state.usar_contextualizacao_ia = (modo_uso.startswith("Pedir para a IA"))

    # Lógica para seleção de parágrafos
    if not st.session_state.usar_contextualizacao_ia:
        paragrafos = [p.strip() for p in st.session_state.texto_fonte.split("\n") if len(p.strip()) >= 150]
        if paragrafos:
            opcoes_paragrafos = st.multiselect(
                "Selecione os parágrafos que formarão o texto-base:",
                options=paragrafos,
                format_func=lambda p: textwrap.shorten(p, width=150, placeholder="...")
            )
            if opcoes_paragrafos:
                st.session_state.trecho_para_prompt = "\n\n".join(opcoes_paragrafos)
        else:
            st.warning("Não foram encontrados parágrafos longos (>150 caracteres). O texto inteiro será usado como referência.")
            st.session_state.trecho_para_prompt = st.session_state.texto_fonte
    else:
        st.info("A IA usará o documento inteiro para criar um novo texto-base.")
        st.session_state.trecho_para_prompt = st.session_state.texto_fonte

    st.subheader("3.2. Preenchimento da Encomenda ENADE")
    with st.form("encomenda_form"):
        col_fonte1, col_fonte2 = st.columns(2)
        with col_fonte1:
            st.session_state.fonte_info['source'] = st.text_input("Fonte/Veículo", placeholder="Ex: G1, Consultor Jurídico, Nome do Livro")
        with col_fonte2:
            st.session_state.fonte_info['year'] = st.text_input("Ano de Publicação", placeholder="Ex: 2024")

        st.markdown("---")

        col_form1, col_form2 = st.columns(2)
        with col_form1:
            tipo_item = st.selectbox("Tipo de item", ["Múltipla Escolha", "Asserção-Razão", "Discursivo"])
            perfil = st.text_input("Perfil do egresso", placeholder="Ex: Ético, crítico e reflexivo")
            competencia = st.text_input("Competência a ser avaliada", placeholder="Ex: Analisar e solucionar conflitos de natureza ética")
        with col_form2:
            objeto_conhecimento = st.text_input("Objeto de conhecimento", placeholder="Ex: Legislação sobre arbitragem e ética profissional")
            dificuldade = st.select_slider("Nível de Dificuldade", ["Fácil", "Média", "Difícil"], value="Média")
            info_adicional = st.text_area("Instrução adicional para a IA (opcional)", placeholder="Ex: Foque no conflito entre inovação e deveres legais.")

        submitted = st.form_submit_button("🚀 Gerar Questão ENADE")

        if submitted:
            # Validação
            if not st.session_state.fonte_info['source'] or not st.session_state.fonte_info['year']:
                st.error("Por favor, preencha os campos 'Fonte/Veículo' e 'Ano de Publicação'.")
            else:
                with st.spinner(f"Aguarde... O especialista do ENADE ({modelo_ia}) está elaborando sua questão."):
                    # Construção do Prompt Final
                    hoje = datetime.now()
                    meses = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
                    data_acesso = f"{hoje.day:02d} de {meses[hoje.month-1]}. de {hoje.year}"
                    
                    fonte_info_local = st.session_state.fonte_info
                    fonte_str = f"Fonte: {fonte_info_local.get('source', 'Fonte desconhecida')}, {fonte_info_local.get('year', 's.d.')}. Disponível em: <{fonte_info_local.get('link', 's.l.')}>. Acesso em: {data_acesso}."
                    
                    # Adapta a instrução do prompt com base na escolha do usuário
                    if st.session_state.usar_contextualizacao_ia:
                        instrucao_texto_base = f"""**1. DOCUMENTO DE REFERÊNCIA (use-o para criar a questão):**
                        "{st.session_state.trecho_para_prompt}"
                        
                        **INSTRUÇÃO ESPECIAL:** A partir do DOCUMENTO DE REFERÊNCIA acima, sua primeira tarefa é criar um novo **Texto-Base** conciso, objetivo e perfeitamente adequado para uma questão do ENADE. Este novo Texto-Base deve ser o ponto de partida para a questão. Em seguida, elabore a questão completa (enunciado e alternativas) com base no Texto-Base que você criou.
                        """
                    else:
                        instrucao_texto_base = f"""**1. TEXTO-BASE LITERAL (use-o exatamente como está na questão):**
                        "{st.session_state.trecho_para_prompt}"
                        """
                    
                    prompt_final = f"""
                    **ENCOMENDA DE ITEM PARA O ENADE**

                    {instrucao_texto_base}
                    (A referência da fonte para a questão deve ser: {fonte_str})

                    **2. DADOS DA ENCOMENDA:**
                    - **Curso:** {curso_escolhido}
                    - **Tipo de item a ser elaborado:** {tipo_item}
                    - **Perfil do egresso esperado:** {perfil}
                    - **Competência a ser avaliada:** {competencia}
                    - **Objeto de conhecimento principal:** {objeto_conhecimento}
                    - **Nível de dificuldade:** {dificuldade}
                    - **Instrução adicional para a IA:** {info_adicional}

                    **3. TAREFA FINAL:**
                    Elabore a questão completa e pronta para uma avaliação impressa, contendo:
                    - Texto-base (seja o que você criou ou o literal fornecido) com a referência completa no formato ABNT simplificado.
                    - Um enunciado claro e objetivo.
                    - Cinco alternativas (A, B, C, D, E).
                    - Ao final, fora da questão, indique o gabarito no formato: "Gabarito: Letra X".
                    """
                    
                    st.session_state.questao_gerada = gerar_questao_com_llm(prompt_final, modelo_ia, api_key)

# --- ETAPA 4: RESULTADO ---
st.header("Etapa 4: Questão Gerada")

if st.session_state.questao_gerada:
    st.markdown(st.session_state.questao_gerada)

    st.download_button(
        label="📥 Baixar Questão (.txt)",
        data=st.session_state.questao_gerada,
        file_name=f"questao_enade_{curso_escolhido.replace(' ', '_')}_{assunto.replace(' ', '_')[:20]}.txt",
        mime="text/plain"
    )
else:
    st.info("Preencha as etapas anteriores para que a questão gerada apareça aqui.")
