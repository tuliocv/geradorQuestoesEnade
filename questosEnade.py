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
    else:
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
    try:
        if upload.type == "application/pdf":
            reader = PyPDF2.PdfReader(BytesIO(upload.read()))
            return "".join(p.extract_text() or "" for p in reader.pages)
        elif upload.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(upload.read()))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
    return None

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
        else:
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json" if use_json else "text/plain"
            )
            model_gemini = genai.GenerativeModel(model)
            full_prompt = "\n".join(f"**{m['role']}**: {m['content']}" for m in prompt_messages)
            resp = model_gemini.generate_content(full_prompt, generation_config=generation_config)
            return resp.text
    except Exception as e:
        st.error(f"Erro na chamada à API do {provider}: {e}")
    return None

# --- INICIALIZAÇÃO E LAYOUT DO APP ---
st.title("🎓 Gerador de Questões ENADE v2.1")

# Estado inicial
if "text_base" not in st.session_state:
    st.session_state.text_base = ""
if "questao_gerada" not in st.session_state:
    st.session_state.questao_gerada = None
if "lista_questoes" not in st.session_state:
    st.session_state.lista_questoes = []

# --- 1. Definição do Escopo ---
with st.container():
    st.header("1. Definição do Escopo")
    c1, c2, c3 = st.columns(3)
    area_selecionada = c1.selectbox("Área do conhecimento", list(AREAS_ENADE.keys()))
    curso_selecionado = c2.selectbox("Curso", AREAS_ENADE[area_selecionada])
    assunto = c3.text_input("Tópico / Assunto central", placeholder="Ex: IA na arbitragem")
    st.session_state.escopo = {
        "area": area_selecionada,
        "curso": curso_selecionado,
        "assunto": assunto
    }

# --- 2. Inserção do Texto-Base (Opcional) ---
with st.container():
    st.header("2. Inserção do Texto-Base (Opcional)")
    inserir_tb = st.radio(
        "Você deseja inserir um texto-base?",
        ["Não, deixar IA gerar automaticamente", "Sim, vou inserir um texto-base"],
        horizontal=True
    )

    if inserir_tb.startswith("Sim"):
        metodo_tb = st.radio(
            "Como você quer fornecer o texto-base?",
            ["Upload de PDF para IA resumir", "Buscar na Internet (URL)"],
            horizontal=True
        )

        if metodo_tb.startswith("Upload"):
            upload_tb = st.file_uploader(
                "Envie um PDF para resumirmos",
                type=["pdf"]
            )
            if upload_tb:
                with st.spinner("Resumindo PDF..."):
                    texto_pdf = extrair_texto_upload(upload_tb)
                    if texto_pdf:
                        prompt = [
                            {"role":"system","content":"Você cria resumos concisos para questões ENADE."},
                            {"role":"user","content":
                                f"Resuma este texto em até 3 frases, "
                                f"focando nos principais pontos para uma situação-problema ENADE:\n\n{texto_pdf}"
                            }
                        ]
                        st.session_state.text_base = chamar_llm(
                            prompt, provider=provedor_ia, model=model,
                            temperature=0.4, max_tokens=250
                        )
                        st.success("Resumo gerado!")
        else:
            url_tb = st.text_input("Cole aqui a URL do conteúdo")
            if st.button("▶️ Extrair e resumir URL"):
                with st.spinner("Processando URL..."):
                    texto_web, titulo_web, autor_web = extrair_conteudo_url(url_tb)
                    if texto_web:
                        prompt = [
                            {"role":"system","content":"Você cria resumos concisos para questões ENADE."},
                            {"role":"user","content":
                                f"Resuma este texto em até 3 frases, "
                                f"focando nos principais pontos para uma situação-problema ENADE:\n\n{texto_web}"
                            }
                        ]
                        st.session_state.text_base = chamar_llm(
                            prompt, provider=provedor_ia, model=model,
                            temperature=0.4, max_tokens=250
                        )
                        st.session_state.fonte_info = {
                            "titulo": titulo_web,
                            "autor": autor_web,
                            "veiculo": url_tb.split("/")[2],
                            "link": url_tb
                        }
                        st.success("Resumo gerado!")
    else:
        st.info("A IA gerará um texto-base automaticamente depois.")

# --- 3. Texto-Base Final e Referência ---
if st.session_state.text_base:
    with st.container():
        st.header("3. Texto-Base Final e Referência")
        st.session_state.text_base = st.text_area(
            "Texto-Base (edite se quiser)", st.session_state.text_base, height=200
        )
        info = st.session_state.get("fonte_info", {})
        c1, c2, c3, c4 = st.columns(4)
        autor_ref = c1.text_input("Autor (SOBRENOME, Nome)", value=info.get("autor", ""))
        titulo_ref = c2.text_input("Título", value=info.get("titulo", ""))
        veiculo_ref = c3.text_input("Veículo", value=info.get("veiculo", ""))
        data_pub_ref = c4.text_input("Data de publicação", placeholder="dd mmm. aaaa")
        if all([autor_ref, titulo_ref, veiculo_ref, data_pub_ref]):
            hoje = datetime.now()
            meses = ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]
            acesso = f"{hoje.day} {meses[hoje.month-1]} {hoje.year}"
            referencia_abnt = (
                f"{autor_ref}. {titulo_ref}. {veiculo_ref}, {data_pub_ref}. "
                f"Disponível em: <{info.get('link','N/D')}>. Acesso em: {acesso}."
            )
            st.text_area("Referência ABNT:", referencia_abnt, height=100, key="referencia_final")

# --- 4. Geração da Questão ---
if st.session_state.get("texto_base") and st.session_state.get("referencia_final"):
    with st.container():
        st.header("4. Parâmetros e Geração da Questão")
        with st.form("gen_form"):
            perfil = st.text_input("Perfil do egresso")
            competencia = st.text_input("Competência")
            nivel_bloom = st.select_slider("Nível Cognitivo", options=BLOOM_LEVELS, value="Analisar")
            verbos_sug = BLOOM_VERBS[nivel_bloom]
            verbos_sel = st.multiselect("Verbos de Comando", verbos_sug, default=verbos_sug[:2])
            obs = st.text_area("Observações (opcional)")
            gerar = st.form_submit_button("🚀 Gerar Questão")
        if gerar:
            with st.spinner("Gerando..."):
                system_prompt = """
Você é docente especialista do INEP. Crie uma questão padrão ENADE com:
1. Contextualização.
2. Enunciado usando verbos de Bloom.
3. 5 alternativas (A–E), 1 correta.
4. Gabarito.
5. Justificativas breves para cada alternativa.
Linguagem formal e impessoal.
"""
                user_prompt = f"""
Área: {st.session_state.escopo['area']}
Curso: {st.session_state.escopo['curso']}
Assunto: {st.session_state.escopo['assunto']}
Perfil: {perfil}
Competência: {competencia}
Verbos: {', '.join(verbos_sel)}
Observações: {obs}

TEXTO-BASE:
{st.session_state.text_base}

REFERÊNCIA:
{st.session_state.referencia_final}

Saída em JSON com campos:
"contextualizacao","enunciado","alternativas","gabarito","justificativas"
"""
                raw = chamar_llm(
                    [{"role":"system","content":system_prompt},
                     {"role":"user","content":user_prompt}],
                    provider=provedor_ia, model=model,
                    temperature=0.5, max_tokens=1500, use_json=True
                )
                if raw:
                    try:
                        cleaned = raw.strip().replace("```json", "").replace("```", "")
                        q = json.loads(cleaned)
                        st.session_state.questao_gerada = q
                        # adicionar à lista
                        registro = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "area": st.session_state.escopo["area"],
                            "curso": st.session_state.escopo["curso"],
                            "assunto": st.session_state.escopo["assunto"],
                            "contextualizacao": q["contextualizacao"],
                            "enunciado": q["enunciado"],
                            "gabarito": q["gabarito"]
                        }
                        # alternativas e justificativas
                        for lt in ["A","B","C","D","E"]:
                            registro[f"alt_{lt}"] = q["alternativas"].get(lt, "")
                            registro[f"just_{lt}"] = q["justificativas"].get(lt, "")
                        st.session_state.lista_questoes.append(registro)
                    except Exception:
                        st.error("Não foi possível processar a resposta da IA.")

# --- 5. Exibição e Download ---
if st.session_state.questao_gerada:
    q = st.session_state.questao_gerada
    st.header("5. Resultado")
    st.markdown(f"**Contextualização:**  \n{q.get('contextualizacao','')}")
    st.markdown(f"**Enunciado:**  \n{q.get('enunciado','')}")
    st.subheader("Alternativas")
    for lt, txt in q["alternativas"].items():
        st.markdown(f"**{lt}.** {txt}")
    st.success(f"**Gabarito:** {q.get('gabarito','')}")
    with st.expander("Justificativas"):
        for lt, j in q["justificativas"].items():
            st.markdown(f"**{lt}.** {j}")

    # preparar texto para download
    text_content = []
    text_content.append("CONTEXTUALIZAÇÃO:\n" + q["contextualizacao"])
    text_content.append("\nENUNCIADO:\n" + q["enunciado"])
    text_content.append("\nALTERNATIVAS:")
    for lt, txt in q["alternativas"].items():
        text_content.append(f"{lt}. {txt}")
    text_content.append("\nGABARITO: " + q["gabarito"])
    text_content.append("\nJUSTIFICATIVAS:")
    for lt, j in q["justificativas"].items():
        text_content.append(f"{lt}. {j}")
    text_str = "\n".join(text_content)

    c1, c2 = st.columns(2)
    c1.download_button(
        "📄 Baixar Questão (TXT)",
        data=text_str,
        file_name=f"questao_{len(st.session_state.lista_questoes)}.txt",
        mime="text/plain"
    )

    # se houver pelo menos uma questão na lista, oferecer download de Excel
    if st.session_state.lista_questoes:
        df_all = pd.DataFrame(st.session_state.lista_questoes)
        towrite = BytesIO()
        df_all.to_excel(towrite, index=False, sheet_name="Questões")
        towrite.seek(0)
        c2.download_button(
            "📥 Baixar Todas as Questões (.xlsx)",
            data=towrite,
            file_name="todas_questoes_enade.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
