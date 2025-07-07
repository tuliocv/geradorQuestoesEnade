import streamlit as st
import requests
import textwrap
import json
from datetime import datetime
from io import BytesIO
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
import PyPDF2
from docx import Document
import pandas as pd
import os

# --- MODELO DE DADOS E CONSTANTES ---
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

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Gerador de Questões ENADE v2.1", page_icon="🎓", layout="wide")

# --- CONFIGURAÇÃO DAS APIS NA BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuração da IA")
    provedor_ia = st.selectbox("Escolha o Provedor de IA", ["OpenAI (GPT)", "Google (Gemini)"])
    
    api_key = None
    model = None

    if provedor_ia == "OpenAI (GPT)":
        api_key = st.text_input("Sua Chave de API da OpenAI", type="password")
        model = st.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], help="gpt-4o-mini é o mais rápido e barato.")
    elif provedor_ia == "Google (Gemini)":
        api_key = st.text_input("Sua Chave de API do Google AI", type="password")
        model = st.selectbox("Modelo Gemini", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"], help="Flash é mais rápido e barato.")

    st.header("Sobre o App")
    st.info("Esta aplicação utiliza IA para gerar questões no padrão ENADE. Escolha o provedor, insira sua chave e siga as etapas.")

if not api_key:
    st.warning(f"Informe sua chave do {provedor_ia} na barra lateral para habilitar o aplicativo.")
    st.stop()

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600)
def extrair_conteudo_url(url: str):
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        titulo = soup.title.string if soup.title else ""
        autor_meta = soup.find("meta", attrs={"name": "author"})
        autor = autor_meta['content'] if autor_meta else ""
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
        texto = " ".join(soup.stripped_strings)
        return texto, titulo, autor
    except Exception as e:
        st.error(f"Erro ao extrair conteúdo da URL: {e}"); return None, None, None

def extrair_texto_upload(upload):
    try:
        if upload.type == "application/pdf":
            reader = PyPDF2.PdfReader(BytesIO(upload.read()))
            return "".join(p.extract_text() or "" for p in reader.pages)
        elif upload.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(upload.read()))
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}"); return None

def chamar_llm(prompt_messages, provider, model, temperature=0.7, max_tokens=2000, use_json=False):
    try:
        if provider == "OpenAI (GPT)":
            client = OpenAI(api_key=api_key)
            response_format = {"type": "json_object"} if use_json else {"type": "text"}
            resp = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            return resp.choices[0].message.content.strip()
        
        elif provider == "Google (Gemini)":
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json" if use_json else "text/plain"
            )
            model_gemini = genai.GenerativeModel(model)
            # Concatena mensagens para o formato do Gemini
            full_prompt = "\n".join([f"**{m['role']}**: {m['content']}" for m in prompt_messages])
            resp = model_gemini.generate_content(full_prompt, generation_config=generation_config)
            return resp.text
            
    except Exception as e:
        st.error(f"Erro na chamada à API do {provider}: {e}"); return None

# --- INICIALIZAÇÃO E LAYOUT DO APP ---
st.title("🎓 Gerador de Questões ENADE v2.1")

if "full_text" not in st.session_state: st.session_state.full_text = ""
if "text_base" not in st.session_state: st.session_state.text_base = ""
if "questao_gerada" not in st.session_state: st.session_state.questao_gerada = None

with st.container(border=True):
    st.header("1. Definição do Escopo")
    c1, c2, c3 = st.columns(3)
    area_selecionada = c1.selectbox("Área do conhecimento", list(AREAS_ENADE.keys()))
    curso_selecionado = c2.selectbox("Curso", AREAS_ENADE[area_selecionada])
    assunto = c3.text_input("Tópico / Assunto central", placeholder="Ex: IA na arbitragem")
    st.session_state.escopo = {"area": area_selecionada, "curso": curso_selecionado, "assunto": assunto}

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

with st.container(border=True):
    st.header("3. Definição do Trecho-Base e Referência")
    if st.session_state.full_text:
        modo_tb = st.radio("Como obter o trecho para a questão?", ["Selecionar manualmente", "Resumo automático (via IA)"], horizontal=True)
        
        if modo_tb == "Selecionar manualmente":
            paras = [p.strip() for p in st.session_state.full_text.split('\n') if len(p.strip()) > 100]
            sel = st.multiselect("Escolha os parágrafos:", paras, format_func=lambda p: textwrap.shorten(p, 150, placeholder="…"))
            if sel: st.session_state.text_base = "\n\n".join(sel)
        else:
            if st.button("🔎 Gerar resumo automático"):
                prompt = [{"role": "system", "content": "Você cria resumos concisos para questões ENADE."},
                          {"role": "user", "content": f"Resuma o texto a seguir em até 3 frases, focando nos pontos principais para uma situação-problema ENADE:\n\n{st.session_state.full_text}"}]
                with st.spinner("Resumindo o texto..."):
                    st.session_state.text_base = chamar_llm(prompt, provider=provedor_ia, model=model, temperature=0.4, max_tokens=250)

        if st.session_state.text_base:
            st.text_area("Texto-Base final (edite se necessário):", value=st.session_state.text_base, height=150, key="text_base_final")

        info = st.session_state.get("fonte_info", {})
        c1, c2, c3, c4 = st.columns(4)
        autor_ref = c1.text_input("Autor (SOBRENOME, Nome)", value=info.get("autor", ""))
        titulo_ref = c2.text_input("Título", value=info.get("titulo", ""))
        veiculo_ref = c3.text_input("Veículo", value=info.get("veiculo", ""))
        data_pub_ref = c4.text_input("Data de publicação", placeholder="dd mmm. aaaa")
        
        if all([autor_ref, titulo_ref, veiculo_ref, data_pub_ref]):
            hoje, meses = datetime.now(), ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]
            acesso = f"{hoje.day} {meses[hoje.month-1]}. {hoje.year}"
            referencia_abnt = f"{autor_ref}. {titulo_ref}. {veiculo_ref}, {data_pub_ref}. Disponível em: <{info.get('link', 'N/D')}>. Acesso em: {acesso}."
            st.text_area("Referência final:", value=referencia_abnt, height=100, key="referencia_final")

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
                with st.spinner(f"Aguarde, o especialista do ENADE ({provedor_ia} - {model}) está trabalhando..."):
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
                    raw_response = chamar_llm([{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}], provider=provedor_ia, model=model, temperature=0.5, use_json=True)
                    
                    if raw_response:
                        try:
                            # Gemini pode retornar o JSON dentro de um bloco de código markdown
                            cleaned_response = raw_response.strip().replace("```json", "").replace("```", "")
                            st.session_state.questao_gerada = json.loads(cleaned_response)
                        except json.JSONDecodeError:
                            st.error("A IA retornou uma resposta em formato inválido. Tente novamente.")
                            st.expander("Ver resposta bruta da IA").write(raw_response)
                            st.session_state.questao_gerada = None
    else:
        st.info("Preencha as etapas anteriores para habilitar a geração.")

if st.session_state.questao_gerada:
    st.header("5. Resultado e Ações")
    q = st.session_state.questao_gerada
    
    st.markdown(f"**Contextualização:**\n{q.get('contextualizacao', 'N/A')}")
    st.markdown(f"**Enunciado:**\n{q.get('enunciado', 'N/A')}")
    st.subheader("Alternativas")
    for letra, texto in q.get('alternativas', {}).items(): st.markdown(f"**{letra}.** {texto}")
    st.success(f"**Gabarito:** {q.get('gabarito', 'N/A')}")
    
    with st.expander("Ver Justificativas"):
        for letra, just in q.get('justificativas', {}).items(): st.markdown(f"**{letra}.** {just}")

    st.subheader("Ações")
    c1, c2, c3 = st.columns(3)

    doc = Document()
    doc.add_heading(f"Questão ENADE - {st.session_state.escopo['curso']}", level=1)
    doc.add_heading("Texto-Base", level=2); doc.add_paragraph(st.session_state.text_base_final); doc.add_paragraph(st.session_state.referencia_final)
    doc.add_heading("Contextualização", level=2); doc.add_paragraph(q.get('contextualizacao', ''))
    doc.add_heading("Enunciado", level=2); doc.add_paragraph(q.get('enunciado', ''))
    doc.add_heading("Alternativas", level=2)
    for letra, texto in q.get('alternativas', {}).items(): doc.add_paragraph(f"{letra}. {texto}")
    doc.add_paragraph(f"\n**Gabarito:** {q.get('gabarito', '')}")
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    c1.download_button("📥 Baixar em Word (.docx)", data=buffer, file_name="questao_enade.docx")

    if c2.button("💾 Salvar no Banco de Questões"):
        nova_entrada = pd.DataFrame([{"data": datetime.now().strftime("%Y-%m-%d %H:%M"), "curso": st.session_state.escopo['curso'], "assunto": st.session_state.escopo['assunto'], "questao_json": json.dumps(q, ensure_ascii=False)}])
        db_path = "banco_questoes.csv"
        banco_df = pd.read_csv(db_path) if os.path.exists(db_path) else pd.DataFrame()
        banco_df = pd.concat([banco_df, nova_entrada], ignore_index=True)
        banco_df.to_csv(db_path, index=False)
        st.success("Questão salva no banco de dados local!")

    if c3.button("🔄 Gerar Nova Questão (Limpar Tudo)"):
        for key in list(st.session_state.keys()):
            if key not in ['escopo']: del st.session_state[key]
        st.rerun()

if os.path.exists("banco_questoes.csv"):
    with st.expander("📖 Ver Banco de Questões Salvas"):
        df_banco = pd.read_csv("banco_questoes.csv")
        st.dataframe(df_banco)
