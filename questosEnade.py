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
        st.error(f"Não foi possível extrair a URL: {e}")
        return None, None, None

def extrair_texto_upload(upload):
    try:
        if upload.type == "application/pdf":
            reader = PyPDF2.PdfReader(BytesIO(upload.read()))
            return "".join(p.extract_text() or "" for p in reader.pages)
        elif upload.type.startswith("application/"):
            doc = Document(BytesIO(upload.read()))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
    return None

def chamar_llm(prompts, provider, model, temperature=0.7, max_tokens=2000, use_json=False):
    try:
        if provider.startswith("OpenAI"):
            client = OpenAI(api_key=st.session_state.api_key)
            fmt = {"type":"json_object"} if use_json else {"type":"text"}
            r = client.chat.completions.create(
                model=model, messages=prompts,
                temperature=temperature, max_tokens=max_tokens,
                response_format=fmt
            )
            return r.choices[0].message.content.strip()
        else:
            genai.configure(api_key=st.session_state.api_key)
            cfg = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json" if use_json else "text/plain"
            )
            m = genai.GenerativeModel(model)
            full = "\n".join(f"**{m['role']}**: {m['content']}" for m in prompts)
            resp = m.generate_content(full, generation_config=cfg)
            return resp.text
    except Exception as e:
        st.error(f"Erro na API {provider}: {e}")
    return None

def search_articles(query, num=5):
    """Busca rápida no Google e retorna lista de {title, url}."""
    headers = {"User-Agent":"Mozilla/5.0"}
    params = {"q": query, "num": num}
    r = requests.get("https://www.google.com/search", headers=headers, params=params, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    res = []
    for g in soup.select("div.g")[:num]:
        a = g.find("a", href=True)
        h3 = g.find("h3")
        if a and h3:
            res.append({"title": h3.get_text(), "url": a["href"]})
    return res

# --- CONFIG STREAMLIT ---
st.set_page_config("Gerador de Questões ENADE v2.1", "🎓", "wide")

# --- SIDEBAR IA ---
with st.sidebar:
    st.header("⚙️ Configuração IA")
    prov = st.selectbox("Provedor", ["OpenAI (GPT)", "Google (Gemini)"])
    st.session_state.api_key = st.text_input("Chave de API", type="password")
    if prov.startswith("OpenAI"):
        mdl = st.selectbox("Modelo GPT", ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"])
    else:
        mdl = st.selectbox("Modelo Gemini", ["gemini-1.5-flash-latest","gemini-1.5-pro-latest"])
    st.write("---")
    st.info("Gere questões ENADE a partir de texto-base ou deixe a IA criar automaticamente.")

if not st.session_state.api_key:
    st.warning("Informe a chave de API na lateral para continuar.")
    st.stop()

# --- ESTADO INICIAL ---
if "text_base" not in st.session_state:    st.session_state.text_base = ""
if "questoes"  not in st.session_state:    st.session_state.questoes = []
if "search_results" not in st.session_state: st.session_state.search_results = []

# --- 1. ESCOPO ---
st.title("🎓 Gerador de Questões ENADE")
with st.container():
    st.header("1. Definição do Escopo")
    c1,c2,c3 = st.columns(3)
    area  = c1.selectbox("Área", list(AREAS_ENADE.keys()))
    curso = c2.selectbox("Curso", AREAS_ENADE[area])
    assunto = c3.text_input("Assunto central")
    st.session_state.escopo = {"area":area,"curso":curso,"assunto":assunto}

# --- 2. TEXTO-BASE OPCIONAL ---
with st.container():
    st.header("2. Texto-Base (Opcional)")
    escolha = st.radio("Deseja inserir um texto-base?", 
                      ["Não, IA gera automaticamente","Sim, inserir texto-base"], horizontal=True)

    # 2.a) NÃO: gera automático
    if escolha.startswith("Não"):
        if not st.session_state.get("auto"):
            with st.spinner("Gerando texto-base automaticamente..."):
                prompts = [
                    {"role":"system","content":"Você cria trechos-base concisos para questões ENADE."},
                    {"role":"user","content":
                        f"Gere um texto-base de ~3 frases para uma situação-problema ENADE, "
                        f"com base em Área: {area}, Curso: {curso} e Assunto: {assunto}."
                    }
                ]
                tb = chamar_llm(prompts, prov, mdl, temperature=0.5, max_tokens=300)
                st.session_state.text_base = tb or ""
                st.session_state.auto = True
        st.success("Texto-base gerado pela IA!")
    
    # 2.b) SIM: escolher fonte
    else:
        modo = st.radio("Como fornecer o texto-base?", ["Upload de PDF","Buscar artigos na internet"], horizontal=True)
        
        if modo=="Upload de PDF":
            up = st.file_uploader("Envie um PDF para resumir", type="pdf")
            if up:
                with st.spinner("Resumindo PDF..."):
                    txt = extrair_texto_upload(up)
                    if txt:
                        prompts = [
                            {"role":"system","content":"Você cria resumos concisos para ENADE."},
                            {"role":"user","content":
                                f"Resuma em até 3 frases para situação-problema ENADE:\n\n{txt}"
                            }
                        ]
                        st.session_state.text_base = chamar_llm(prompts, prov, mdl, temperature=0.4, max_tokens=250)
                        st.success("Resumo do PDF pronto!")
        
        else:  # buscar artigos
            if st.button("🔍 Buscar artigos sobre o assunto"):
                with st.spinner("Buscando artigos..."):
                    st.session_state.search_results = search_articles(assunto)
            if st.session_state.search_results:
                opts = [f"{a['title']}  ({a['url']})" for a in st.session_state.search_results]
                sel = st.selectbox("Selecione um artigo:", opts)
                if st.button("▶️ Usar artigo selecionado"):
                    idx = opts.index(sel)
                    art = st.session_state.search_results[idx]
                    with st.spinner("Extraindo e resumindo artigo..."):
                        cont, tit, aut = extrair_conteudo_url(art["url"])
                        if cont:
                            prompts = [
                                {"role":"system","content":"Você cria resumos concisos para ENADE."},
                                {"role":"user","content":
                                    f"Resuma em até 3 frases para situação-problema ENADE:\n\n{cont}"
                                }
                            ]
                            st.session_state.text_base = chamar_llm(prompts, prov, mdl, temperature=0.4, max_tokens=250)
                            st.session_state.fonte_info = {"titulo":tit,"autor":aut,"veiculo":art["url"].split("/")[2],"link":art["url"]}
                            st.success("Resumo do artigo pronto!")

# --- 3. EDITAR TEXTO-BASE E REFERÊNCIA ---
if st.session_state.text_base:
    st.header("3. Texto-Base e Referência")
    st.session_state.text_base = st.text_area("Texto-Base (edite se quiser):", st.session_state.text_base, height=200)
    # Referência manual
    info = st.session_state.get("fonte_info",{})
    c1,c2,c3,c4 = st.columns(4)
    autor = c1.text_input("Autor", value=info.get("autor",""))
    tit   = c2.text_input("Título", value=info.get("titulo",""))
    veh   = c3.text_input("Veículo", value=info.get("veiculo",""))
    data  = c4.text_input("Data pub.", placeholder="dd mmm. aaaa")
    if autor and tit and veh and data:
        hoje = datetime.now()
        meses = ["jan.","fev.","mar.","abr.","mai.","jun.","jul.","ago.","set.","out.","nov.","dez."]
        acc = f"{hoje.day} {meses[hoje.month-1]}. {hoje.year}"
        ref = f"{autor}. {tit}. {veh}, {data}. Disponível em: <{info.get('link','')}>. Acesso em: {acc}."
        st.text_area("Referência ABNT:", ref, height=80, key="ref_final")

# --- 4. GERAÇÃO DA QUESTÃO ---
if st.session_state.get("text_base") and st.session_state.get("ref_final"):
    st.header("4. Parâmetros e Geração")
    with st.form("frm"):
        perfil = st.text_input("Perfil do egresso")
        comp   = st.text_input("Competência")
        niv    = st.select_slider("Nível Bloom", options=BLOOM_LEVELS, value="Analisar")
        verbs  = st.multiselect("Verbos de comando", BLOOM_VERBS[niv], default=BLOOM_VERBS[niv][:2])
        obs    = st.text_area("Observações (opcional)")
        bot   = st.form_submit_button("🚀 Gerar")
    if bot:
        with st.spinner("Gerando questão..."):
            sys_p = """
Você é docente especialista do INEP. Gere UMA questão ENADE em texto puro, no formato:
CONTEXTUALIZAÇÃO:
<...>
ENUNCIADO:
<...>
ALTERNATIVAS:
A. ...
B. ...
C. ...
D. ...
E. ...
GABARITO:
Letra X
JUSTIFICATIVAS:
A. ...
B. ...
C. ...
D. ...
E. ...
"""
            usr_p = f"""
Área: {area}
Curso: {curso}
Assunto: {assunto}
Perfil: {perfil}
Competência: {comp}
Verbos: {', '.join(verbs)}
Observações: {obs}

TEXTO-BASE:
{st.session_state.text_base}

REFERÊNCIA:
{st.session_state.ref_final}

Por favor, siga EXATAMENTE o formato acima.
"""
            out = chamar_llm([{"role":"system","content":sys_p},
                              {"role":"user","content":usr_p}],
                              prov, mdl, temperature=0.5, max_tokens=1000)
            if out:
                st.session_state.questoes.append(out)
                st.success("Questão gerada!")

# --- 5. RESULTADOS E DOWNLOAD ---
if st.session_state.questoes:
    st.header("5. Questões Geradas")
    for i, q in enumerate(st.session_state.questoes, 1):
        st.markdown(f"---\n**Questão #{i}**\n```\n{q}\n```")
    cols = st.columns(2)
    # TXT da última
    cols[0].download_button("📄 Baixar última (.txt)", "\n\n".join(st.session_state.questoes[-1:]),
                            f"questao_{len(st.session_state.questoes)}.txt", "text/plain")
    # EXCEL de todas
    df = pd.DataFrame({"questão": st.session_state.questoes})
    to_xl = BytesIO()
    df.to_excel(to_xl, index=False, sheet_name="Questões")
    to_xl.seek(0)
    cols[1].download_button("📥 Baixar todas (.xlsx)", to_xl.getvalue(),
                            "todas_questoes_enade.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
