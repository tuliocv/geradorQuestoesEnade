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
import pandas as pd
import os

# --- MODELO DE DADOS E CONSTANTES ---
# Dicionário de áreas e cursos para seleção
AREAS_ENADE = {
    "Ciências Sociais Aplicadas": ["Direito", "Administração", "Ciências Contábeis", "Jornalismo", "Publicidade e Propaganda", "Turismo"],
    "Engenharias": ["Engenharia de Software", "Engenharia Civil", "Engenharia de Produção", "Engenharia Elétrica", "Engenharia Mecânica"],
    "Ciências da Saúde": ["Medicina", "Enfermagem", "Farmácia", "Fisioterapia", "Nutrição"],
    "Ciências Humanas": ["Pedagogia", "História", "Letras", "Psicologia"],
}

# Taxonomia de Bloom para guiar a criação das questões
BLOOM_LEVELS = ["Lembrar", "Compreender", "Aplicar", "Analisar", "Avaliar", "Criar"]
BLOOM_VERBS = {
    "Lembrar": ["definir", "listar", "identificar", "recordar", "nomear", "reconhecer"],
    "Compreender": ["explicar", "resumir", "interpretar", "classificar", "descrever", "discutir"],
    "Aplicar": ["usar", "implementar", "executar", "demonstrar", "resolver", "calcular"],
    "Analisar": ["diferenciar", "organizar", "atribuir", "comparar", "examinar", "categorizar"],
    "Avaliar": ["julgar", "criticar", "justificar", "avaliar", "defender", "recomendar"],
    "Criar": ["projetar", "construir", "formular", "sintetizar", "planejar", "desenvolver"]
}

# --- CONFIGURAÇÃO DA PÁGINA E API KEY ---
st.set_page_config(page_title="Gerador de Questões ENADE v2.0", page_icon="🎓", layout="wide")

with st.sidebar:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input("Insira sua chave da OpenAI aqui", type="password")
    model = st.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], help="gpt-4o-mini é o mais rápido e barato.")
    
    st.header("Sobre o App")
    st.info("Este aplicativo foi atualizado para otimizar a criação de questões padrão ENADE, unificando chamadas de IA, automatizando referências e adicionando novas funcionalidades como análise da questão e banco de dados.")

if not api_key:
    st.warning("Informe sua chave da OpenAI na barra lateral para habilitar o aplicativo.")
    st.stop()

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600)
def extrair_conteudo_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        titulo = soup.title.string if soup.title else ""
        autor_meta = soup.find("meta", attrs={"name": "author"})
        autor = autor_meta['content'] if autor_meta else ""
        
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        texto = " ".join(soup.stripped_strings)
        return texto, titulo, autor
    except Exception as e:
        st.error(f"Erro ao extrair conteúdo da URL: {e}")
        return None, None, None

def extrair_texto_upload(upload):
    if upload.type == "application/pdf":
        try:
            reader = PyPDF2.PdfReader(BytesIO(upload.read()))
            return "".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            st.error(f"Erro ao ler PDF: {e}")
    elif upload.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = Document(BytesIO(upload.read()))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"Erro ao ler DOCX: {e}")
    return None

def chamar_llm(messages, temperature=0.7, max_tokens=1500, use_json=False):
    try:
        client = OpenAI(api_key=api_key)
        response_format = {"type": "json_object"} if use_json else {"type": "text"}
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Erro na chamada à API da OpenAI: {e}")
        return None

# --- INICIALIZAÇÃO E LAYOUT DO APP ---
st.title("🎓 Gerador de Questões ENADE v2.0")

# Inicializa o estado da sessão
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "text_base" not in st.session_state:
    st.session_state.text_base = ""
if "questao_gerada" not in st.session_state:
    st.session_state.questao_gerada = None

# Etapa 1: Definição do Escopo
with st.container(border=True):
    st.header("1. Definição do Escopo")
    c1, c2, c3 = st.columns(3)
    area_selecionada = c1.selectbox("Área do conhecimento", list(AREAS_ENADE.keys()))
    curso_selecionado = c2.selectbox("Curso", AREAS_ENADE[area_selecionada])
    assunto = c3.text_input("Tópico / Assunto central", placeholder="Ex: IA na arbitragem")
    st.session_state.escopo = {"area": area_selecionada, "curso": curso_selecionado, "assunto": assunto}

# Etapa 2: Carregar Texto-Base
with st.container(border=True):
    st.header("2. Fornecimento do Texto-Base")
    metodo = st.radio("Origem do texto:", ["URL", "Upload de Arquivo"], horizontal=True, key="metodo_upload")
    
    if metodo == "URL":
        url = st.text_input("Cole a URL completa do artigo ou notícia")
        if st.button("▶️ Extrair da URL"):
            texto, titulo, autor = extrair_conteudo_url(url)
            if texto:
                st.session_state.full_text = texto
                st.session_state.fonte_info = {"titulo": titulo, "autor": autor, "veiculo": url.split('/')[2], "link": url}
    else:
        upload = st.file_uploader("Envie um arquivo PDF ou DOCX", type=["pdf", "docx"])
        if upload:
            texto = extrair_texto_upload(upload)
            if texto:
                st.session_state.full_text = texto
                st.session_state.fonte_info = {"titulo": "", "autor": "", "veiculo": "", "link": upload.name}

if st.session_state.full_text:
    with st.expander("Ver / Editar Texto Completo Extraído"):
        st.session_state.full_text = st.text_area("Texto", st.session_state.full_text, height=250)

# Etapa 3: Definição do Trecho-Base e Referência
with st.container(border=True):
    st.header("3. Definição do Trecho-Base e Referência")
    if st.session_state.full_text:
        modo_tb = st.radio("Como obter o trecho para a questão?", ["Selecionar manualmente", "Resumo automático (via IA)"], horizontal=True)
        
        if modo_tb == "Selecionar manualmente":
            paras = [p.strip() for p in st.session_state.full_text.split('\n') if len(p.strip()) > 100]
            sel = st.multiselect("Escolha os parágrafos:", paras, format_func=lambda p: textwrap.shorten(p, 150, placeholder="…"))
            if sel:
                st.session_state.text_base = "\n\n".join(sel)
        else:
            if st.button("🔎 Gerar resumo automático"):
                prompt = [{"role": "system", "content": "Você cria resumos concisos para questões ENADE."},
                          {"role": "user", "content": f"Resuma o texto a seguir em até 3 frases, focando nos pontos principais para uma situação-problema ENADE:\n\n{st.session_state.full_text}"}]
                with st.spinner("Resumindo o texto..."):
                    st.session_state.text_base = chamar_llm(prompt, temperature=0.4, max_tokens=250)

        if st.session_state.text_base:
            st.text_area("Texto-Base final (edite se necessário):", value=st.session_state.text_base, height=150, key="text_base_final")

        st.subheader("Referência ABNT (preenchida automaticamente)")
        info = st.session_state.fonte_info
        c1, c2, c3, c4 = st.columns(4)
        autor_ref = c1.text_input("Autor (SOBRENOME, Nome)", value=info.get("autor", ""))
        titulo_ref = c2.text_input("Título", value=info.get("titulo", ""))
        veiculo_ref = c3.text_input("Veículo", value=info.get("veiculo", ""))
        data_pub_ref = c4.text_input("Data de publicação", placeholder="dd mmm. aaaa")
        
        if all([autor_ref, titulo_ref, veiculo_ref, data_pub_ref]):
            hoje, meses = datetime.now(), ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]
            acesso = f"{hoje.day} {meses[hoje.month-1]}. {hoje.year}"
            referencia_abnt = f"{autor_ref}. {titulo_ref}. {veiculo_ref}, {data_pub_ref}. Disponível em: <{info['link']}>. Acesso em: {acesso}."
            st.text_area("Referência final:", value=referencia_abnt, height=100, key="referencia_final")

# Etapa 4: Geração da Questão
with st.container(border=True):
    st.header("4. Parâmetros e Geração da Questão")
    if st.session_state.get("text_base_final") and st.session_state.get("referencia_final"):
        with st.form("generation_form"):
            st.subheader("Parâmetros da Questão (ENADE)")
            perfil = st.text_input("Perfil do egresso", placeholder="Ex: Profissional crítico, ético e reflexivo...")
            competencia = st.text_input("Competência", placeholder="Ex: Avaliar criticamente o impacto de novas tecnologias...")
            
            st.subheader("Taxonomia de Bloom")
            nivel_bloom = st.select_slider("Nível Cognitivo", options=BLOOM_LEVELS, value="Analisar")
            verbos_sugeridos = BLOOM_VERBS[nivel_bloom]
            verbos_selecionados = st.multiselect("Verbos de Comando Sugeridos:", verbos_sugeridos, default=verbos_sugeridos[0] if verbos_sugeridos else None)
            
            observacoes = st.text_area("Observações/Instruções adicionais para a IA (opcional)")

            submit_button = st.form_submit_button("🚀 Gerar Questão Completa")

            if submit_button:
                with st.spinner("Aguarde, o especialista do ENADE está trabalhando..."):
                    system_prompt = """
                    Você é um docente especialista do INEP. Sua tarefa é criar uma questão completa padrão ENADE em uma única etapa, retornando um único JSON.
                    1. Crie uma breve **contextualização (situação-problema)** com base no texto-base e nos parâmetros.
                    2. Com base na contextualização, elabore o **enunciado** usando os verbos de Bloom.
                    3. Crie 5 **alternativas** (A-E), sendo apenas 1 correta. Os distratores devem ser plausíveis e baseados em erros comuns.
                    4. Indique o **gabarito**.
                    5. Forneça **justificativas** breves para CADA alternativa, explicando por que está certa ou errada.
                    Siga rigorosamente as regras: linguagem formal, impessoal, foco em aplicação e sem termos absolutos.
                    """
                    user_prompt = f"""
                    # DADOS PARA GERAÇÃO
                    - Área: {st.session_state.escopo['area']}
                    - Curso: {st.session_state.escopo['curso']}
                    - Assunto: {st.session_state.escopo['assunto']}
                    - Perfil do Egresso: {perfil}
                    - Competência a ser Avaliada: {competencia}
                    - Verbos de Bloom (foco): {', '.join(verbos_selecionados)}
                    - Observações: {observacoes}

                    ## TEXTO-BASE
                    {st.session_state.text_base_final}

                    ## FORMATO DE SAÍDA OBRIGATÓRIO (JSON):
                    {{
                      "contextualizacao": "...",
                      "enunciado": "...",
                      "alternativas": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}},
                      "gabarito": "Letra X",
                      "justificativas": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}}
                    }}
                    """
                    raw_response = chamar_llm([{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}], temperature=0.5, use_json=True)
                    
                    if raw_response:
                        try:
                            st.session_state.questao_gerada = json.loads(raw_response)
                        except json.JSONDecodeError:
                            st.error("A IA retornou uma resposta em formato inválido. Tente novamente.")
                            st.expander("Ver resposta bruta da IA").write(raw_response)
                            st.session_state.questao_gerada = None
    else:
        st.info("Preencha as etapas anteriores para habilitar a geração.")

# Etapa 5: Resultado e Ações
if st.session_state.questao_gerada:
    st.header("5. Resultado da Geração")
    q = st.session_state.questao_gerada
    
    # Exibição da questão na tela
    st.markdown(f"**Contextualização:**\n{q.get('contextualizacao', 'N/A')}")
    st.markdown(f"**Enunciado:**\n{q.get('enunciado', 'N/A')}")
    st.subheader("Alternativas")
    for letra, texto in q.get('alternativas', {}).items():
        st.markdown(f"**{letra}.** {texto}")
    st.success(f"**Gabarito:** {q.get('gabarito', 'N/A')}")
    
    with st.expander("Ver Justificativas"):
        for letra, just in q.get('justificativas', {}).items():
            st.markdown(f"**{letra}.** {just}")

    # Ações com a questão gerada
    st.subheader("Ações")
    c1, c2, c3 = st.columns(3)

    # 1. Download em DOCX
    doc = Document()
    doc.add_heading(f"Questão ENADE - {st.session_state.escopo['curso']}", level=1)
    doc.add_heading("Texto-Base", level=2)
    doc.add_paragraph(st.session_state.text_base_final)
    doc.add_paragraph(st.session_state.referencia_final)
    doc.add_heading("Contextualização", level=2)
    doc.add_paragraph(q.get('contextualizacao', ''))
    doc.add_heading("Enunciado", level=2)
    doc.add_paragraph(q.get('enunciado', ''))
    doc.add_heading("Alternativas", level=2)
    for letra, texto in q.get('alternativas', {}).items():
        doc.add_paragraph(f"{letra}. {texto}")
    doc.add_paragraph(f"\n**Gabarito:** {q.get('gabarito', '')}")
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    c1.download_button("📥 Baixar em Word (.docx)", data=buffer, file_name="questao_enade.docx")

    # 2. Análise da Questão (Feature Avançada)
    if c2.button("🔬 Analisar Questão Gerada"):
        with st.spinner("A IA está analisando a própria questão..."):
            prompt_analise = f"""
            Analise a questão ENADE recém-criada, focando em 3 pontos:
            1.  **Alinhamento com Bloom:** A questão realmente avalia o nível cognitivo de '{nivel_bloom}', usando os verbos '{', '.join(verbos_selecionados)}'? Justifique.
            2.  **Qualidade dos Distratores:** As alternativas incorretas são plausíveis? Elas representam erros conceituais comuns ou são fáceis de descartar?
            3.  **Dificuldade Geral:** A questão parece ser de nível '{submit_button.difficulty if 'difficulty' in locals() else 'Médio'}' para um concluinte do curso de '{st.session_state.escopo['curso']}'?
            
            **Questão para Análise:**
            {json.dumps(q, indent=2, ensure_ascii=False)}
            """
            analise = chamar_llm([{"role":"system","content":"Você é um especialista em avaliação educacional."}, {"role":"user","content":prompt_analise}], temperature=0.3)
            st.session_state.analise_gerada = analise

    # 3. Salvar no Banco de Questões
    if c3.button("💾 Salvar no Banco de Questões"):
        nova_entrada = pd.DataFrame([{
            "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "curso": st.session_state.escopo['curso'],
            "assunto": st.session_state.escopo['assunto'],
            "questao_json": json.dumps(q, ensure_ascii=False)
        }])
        db_path = "banco_questoes.csv"
        if os.path.exists(db_path):
            banco_df = pd.read_csv(db_path)
            banco_df = pd.concat([banco_df, nova_entrada], ignore_index=True)
        else:
            banco_df = nova_entrada
        banco_df.to_csv(db_path, index=False)
        st.success("Questão salva no banco de dados local (`banco_questoes.csv`)!")

if "analise_gerada" in st.session_state:
    st.subheader("Análise da Questão")
    st.markdown(st.session_state.analise_gerada)

# Exibição do Banco de Questões
if os.path.exists("banco_questoes.csv"):
    with st.expander("📖 Ver Banco de Questões Salvas"):
        df_banco = pd.read_csv("banco_questoes.csv")
        st.dataframe(df_banco)
        if st.button("Limpar Histórico de Questões"):
            os.remove("banco_questoes.csv")
            st.rerun()
