import streamlit as st
import requests
from datetime import datetime
from io import BytesIO
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
import PyPDF2
from docx import Document
import pandas as pd

# --- CONSTANTES ---
AREAS_ENADE = {
    "Ciências Sociais Aplicadas": ["Direito", "Administração", "Ciências Contábeis", "Jornalismo", "Publicidade e Propaganda", "Turismo"],
    "Engenharias": ["Engenharia de Software", "Engenharia Civil", "Engenharia de Produção", "Engenharia Elétrica", "Engenharia Mecânica"],
    "Ciências da Saúde": ["Medicina", "Enfermagem", "Farmácia", "Fisioterapia", "Nutrição"],
    "Ciências Humanas": ["Pedagogia", "História", "Letras", "Psicologia"],
}
BLOOM_LEVELS = ["Lembrar", "Compreender", "Aplicar", "Analisar", "Avaliar", "Criar"]
BLOOM_VERBS = {
    "Lembrar": ["definir", "listar", "identificar", "recordar", "nomear", "reconhecer"],
    "Compreender": ["explicar", "resumir", "interpretar", "classificar", "descrever", "discutir"],
    "Aplicar": ["usar", "implementar", "executar", "demonstrar", "resolver", "calcular"],
    "Analisar": ["diferenciar", "organizar", "atribuir", "comparar", "examinar", "categorizar"],
    "Avaliar": ["julgar", "criticar", "justificar", "avaliar", "defender", "recomendar"],
    "Criar": ["projetar", "construir", "formular", "sintetizar", "planejar", "desenvolver"]
}

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Gerador de Questões ENADE v3.0", page_icon="🎓", layout="wide")

# --- ESTADO INICIAL ---
# Inicializa todos os estados necessários para a sessão
st.session_state.setdefault("api_key", None)
st.session_state.setdefault("text_base", "")
st.session_state.setdefault("auto", False)
st.session_state.setdefault("ref_final", "")
st.session_state.setdefault("search_results", [])
st.session_state.setdefault("perfil", "")
st.session_state.setdefault("competencia", "")
# Novos estados para histórico e interatividade
st.session_state.setdefault("questoes_geradas", []) # Lista de dicionários com todos os dados da questão
st.session_state.setdefault("selected_index", 0)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuração IA")
    provedor = st.selectbox("Provedor", ["OpenAI (GPT)", "Google (Gemini)"])
    st.session_state.api_key = st.text_input("Chave de API", type="password", value=st.session_state.api_key)
    if provedor.startswith("OpenAI"):
        modelo = st.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        modelo = st.selectbox("Modelo Gemini", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"])
    st.info("Versão 3.0: Refinamento interativo, análise de qualidade e histórico de sessão.")

    # NOVO: HISTÓRICO DE QUESTÕES NA BARRA LATERAL
    st.header("📜 Histórico da Sessão")
    if not st.session_state.questoes_geradas:
        st.caption("Nenhuma questão gerada nesta sessão.")
    else:
        titulos = [q["titulo"] for q in st.session_state.questoes_geradas]
        selected_title = st.radio(
            "Selecione uma questão para ver/editar:",
            options=titulos,
            index=st.session_state.selected_index,
            key="historico_radio"
        )
        st.session_state.selected_index = titulos.index(selected_title)

if not st.session_state.api_key:
    st.warning("Informe a chave de API na lateral para continuar.")
    st.stop()

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600)
def extrair_conteudo_url(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string if soup.title else ""
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
        return " ".join(soup.stripped_strings), title, ""
    except: return None, None, None

def chamar_llm(prompts, prov, mdl, temperature=0.7, max_tokens=2000):
    # (Função original sem modificações)
    try:
        if prov.startswith("OpenAI"):
            client = OpenAI(api_key=st.session_state.api_key)
            r = client.chat.completions.create(model=mdl, messages=prompts, temperature=temperature, max_tokens=max_tokens, response_format={"type": "text"})
            return r.choices[0].message.content.strip()
        else:
            genai.configure(api_key=st.session_state.api_key)
            cfg = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens, response_mime_type="text/plain")
            m = genai.GenerativeModel(mdl)
            prompt_text = "\n".join(f"**{m['role']}**: {m['content']}" for m in prompts)
            resp = m.generate_content(prompt_text, generation_config=cfg)
            return resp.text
    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
        return None

# --- LAYOUT PRINCIPAL ---
st.title("🎓 Gerador de Questões ENADE v3.0")
st.markdown("Bem-vindo ao gerador interativo. Siga os passos para criar, analisar e refinar suas questões.")

# --- COLUNAS PARA INPUTS E OUTPUTS ---
col_input, col_output = st.columns(2, gap="large")

with col_input:
    st.header("1. Definições da Questão")
    
    with st.container(border=True):
        st.subheader("Escopo e Tipo")
        c1, c2 = st.columns(2)
        area = c1.selectbox("Área", list(AREAS_ENADE.keys()))
        curso = c2.selectbox("Curso", AREAS_ENADE[area])
        assunto = st.text_input("Assunto central", "")
        
        tipos_questao = {
            "Múltipla Escolha Tradicional": "apresentar enunciado + alternativas (1 correta)",
            "Complementação": "frase com lacuna '___', alternativas completam",
            "Afirmação-Razão": "afirmação e razão, avaliar verdade e justificativa",
            "Resposta Múltipla": "selecionar/agrupar várias corretas"
        }
        question_type = st.selectbox("Tipo de questão", options=list(tipos_questao.keys()))

    with st.container(border=True):
        st.subheader("Definição Pedagógica")
        perfil = st.text_input("Perfil do egresso a ser avaliado", st.session_state.perfil, help="Descreva o que se espera do profissional formado.")
        competencia = st.text_input("Competência a ser avaliada", st.session_state.competencia, help="Descreva a habilidade que a questão deve medir.")

    with st.container(border=True):
        st.subheader("Fonte do Texto-Base")
        opc = st.radio("Selecione a fonte:", ["Gerar com IA", "Fornecer um texto-base"], horizontal=True)
        
        if opc == "Gerar com IA":
            st.session_state.auto = True
            if perfil and competencia:
                if st.button("Gerar Contextualização com IA", use_container_width=True):
                    # Lógica de geração de texto-base... (semelhante à anterior)
                    st.session_state.text_base = "Texto gerado pela IA com base no perfil e competência." # Placeholder
            else:
                st.warning("Preencha o Perfil e a Competência para gerar o texto-base.")
        else:
            st.session_state.auto = False
            # Lógica de upload ou busca na web... (semelhante à anterior)
            st.session_state.text_base = st.text_area("Cole aqui seu texto-base ou use as buscas (ainda não implementado nesta versão).", height=150)
            st.session_state.ref_final = st.text_input("Referência ABNT do texto-base (se aplicável):")
    
    with st.container(border=True):
        st.subheader("Parâmetros de Geração")
        with st.form("frm_gerar"):
            niv = st.select_slider("Nível Bloom", options=BLOOM_LEVELS, value="Analisar")
            dificuldade = st.slider("Nível de dificuldade", 1, 5, 3)
            gerar = st.form_submit_button("🚀 Gerar Nova Questão", use_container_width=True, type="primary")

            if gerar:
                if not st.session_state.text_base:
                    st.error("É necessário ter um Texto-Base para gerar a questão.")
                else:
                    with st.spinner("Gerando questão e análise de qualidade... Isso pode levar um momento."):
                        # 1. GERAR A QUESTÃO
                        sys_p_geracao = "Você é um docente especialista em produzir questões no estilo ENADE..." # Prompt de geração
                        usr_p_geracao = f"""
                        GERAR QUESTÃO ENADE:
                        - Área: {area}, Curso: {curso}, Assunto: {assunto}
                        - Perfil: {perfil}, Competência: {competencia}
                        - Tipo: {question_type}, Dificuldade: {dificuldade}/5, Nível Bloom: {niv}
                        - TEXTO-BASE: {st.session_state.text_base}
                        - REFERÊNCIA: {st.session_state.ref_final if not st.session_state.auto else 'Texto gerado por IA.'}
                        - FORMATO DE SAÍDA: ENUNCIADO: ..., ALTERNATIVAS: A..., B..., GABARITO:..., JUSTIFICATIVAS:...
                        """
                        questao_gerada = chamar_llm([{"role": "system", "content": sys_p_geracao}, {"role": "user", "content": usr_p_geracao}], provedor, modelo)

                        # 2. GERAR A ANÁLISE DE QUALIDADE
                        sys_p_analise = """
                        Você é um avaliador de itens do ENADE, um especialista em pedagogia e avaliação. Sua tarefa é fornecer uma análise crítica e construtiva da questão fornecida.
                        Seja direto e objetivo. Use bullet points.
                        AVALIE OS SEGUINTES PONTOS:
                        - **Clareza e Pertinência:** O enunciado é claro? Ele se conecta bem ao texto-base?
                        - **Qualidade dos Distratores:** As alternativas incorretas (distratores) são plausíveis? Elas testam erros conceituais comuns ou são fáceis demais?
                        - **Alinhamento Pedagógico:** A questão realmente avalia a competência, o nível de dificuldade e o nível de Bloom solicitados?
                        - **Potencial de Melhoria:** Dê uma sugestão para melhorar a questão.
                        """
                        analise_qualidade = chamar_llm([{"role": "system", "content": sys_p_analise}, {"role": "user", "content": questao_gerada}], provedor, modelo, temperature=0.3)

                        if questao_gerada and analise_qualidade:
                            novo_item = {
                                "titulo": f"Q{len(st.session_state.questoes_geradas) + 1}: {curso} - {assunto[:25]}...",
                                "texto_completo": questao_gerada,
                                "analise_qualidade": analise_qualidade,
                                # Armazenar contexto para refinamento
                                "contexto": {"area": area, "curso": curso, "assunto": assunto, "perfil": perfil, "competencia": competencia, "texto_base": st.session_state.text_base}
                            }
                            st.session_state.questoes_geradas.append(novo_item)
                            st.session_state.selected_index = len(st.session_state.questoes_geradas) - 1
                            st.success("Questão e análise geradas!")
                            st.rerun() # Atualiza o radio da sidebar

with col_output:
    st.header("2. Análise e Refinamento")

    if not st.session_state.questoes_geradas:
        st.info("A questão gerada, junto com sua análise de qualidade e opções de refinamento, aparecerá aqui.")
    else:
        q_selecionada = st.session_state.questoes_geradas[st.session_state.selected_index]
        
        st.subheader(f"Visualizando: {q_selecionada['titulo']}")

        tab_view, tab_analise, tab_refino = st.tabs(["📝 Questão", "🔍 Análise de Qualidade (IA)", "✨ Refinamento Iterativo (IA)"])

        with tab_view:
            st.text_area("Texto da Questão", value=q_selecionada["texto_completo"], height=500, key=f"q_view_{st.session_state.selected_index}")
            c1, c2 = st.columns(2)
            c1.download_button("📄 Baixar esta questão (.txt)", q_selecionada["texto_completo"], f"{q_selecionada['titulo']}.txt", use_container_width=True)
            
            # Preparar para download de todas
            df_all = pd.DataFrame(st.session_state.questoes_geradas)
            to_xl = BytesIO()
            df_all.to_excel(to_xl, index=False, sheet_name="Questões")
            to_xl.seek(0)
            c2.download_button("📥 Baixar todas (.xlsx)", to_xl, "banco_completo_enade.xlsx", use_container_width=True)

        with tab_analise:
            st.info("Esta análise foi gerada por uma IA especialista para ajudar na validação da questão.")
            st.markdown(q_selecionada["analise_qualidade"])

        with tab_refino:
            st.warning("Ações de refinamento modificarão a questão atual. A versão original será perdida.")
            
            r_c1, r_c2, r_c3 = st.columns(3)

            if r_c1.button("🤔 Tornar Mais Difícil", use_container_width=True):
                instrucao = "Reescreva a questão a seguir para torná-la significativamente mais difícil. Aumente a complexidade do enunciado, torne os distratores mais sutis e exija um nível de raciocínio mais elevado, mantendo o mesmo gabarito."
                with st.spinner("Refinando para aumentar a dificuldade..."):
                    prompt_refino = f"{instrucao}\n\nQUESTÃO ATUAL:\n{q_selecionada['texto_completo']}"
                    texto_refinado = chamar_llm([{"role": "user", "content": prompt_refino}], provedor, modelo)
                    st.session_state.questoes_geradas[st.session_state.selected_index]["texto_completo"] = texto_refinado
                    st.rerun()

            if r_c2.button("✍️ Simplificar o Enunciado", use_container_width=True):
                instrucao = "Reescreva apenas o ENUNCIADO da questão a seguir para torná-lo mais claro, direto e objetivo, sem alterar o nível de dificuldade das alternativas ou o gabarito."
                with st.spinner("Refinando para simplificar o enunciado..."):
                     prompt_refino = f"{instrucao}\n\nQUESTÃO ATUAL:\n{q_selecionada['texto_completo']}"
                     texto_refinado = chamar_llm([{"role": "user", "content": prompt_refino}], provedor, modelo)
                     st.session_state.questoes_geradas[st.session_state.selected_index]["texto_completo"] = texto_refinado
                     st.rerun()

            if r_c3.button("🔄 Regenerar Alternativas", use_container_width=True):
                instrucao = "Mantenha o TEXTO-BASE e o ENUNCIADO da questão a seguir, mas gere um conjunto completamente novo de 5 ALTERNATIVAS, 1 GABARITO e suas respectivas JUSTIFICATIVAS. Crie distratores plausíveis e desafiadores."
                with st.spinner("Regenerando as alternativas..."):
                    prompt_refino = f"{instrucao}\n\nQUESTÃO ATUAL:\n{q_selecionada['texto_completo']}"
                    texto_refinado = chamar_llm([{"role": "user", "content": prompt_refino}], provedor, modelo)
                    st.session_state.questoes_geradas[st.session_state.selected_index]["texto_completo"] = texto_refinado
                    st.rerun()

    # Botão para limpar a sessão
    if st.sidebar.button("🔴 Encerrar e Limpar Sessão", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
