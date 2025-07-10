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
    "Ciências Sociais Aplicadas": [
        "Direito", "Administração", "Ciências Contábeis",
        "Jornalismo", "Publicidade e Propaganda", "Turismo"
    ],
    "Engenharias": [
        "Engenharia de Software", "Engenharia Civil",
        "Engenharia de Produção", "Engenharia Elétrica",
        "Engenharia Mecânica"
    ],
    "Ciências da Saúde": [
        "Medicina", "Enfermagem", "Farmácia",
        "Fisioterapia", "Nutrição"
    ],
    "Ciências Humanas": [
        "Pedagogia", "História", "Letras", "Psicologia"
    ],
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
st.set_page_config(
    page_title="Gerador de Questões ENADE v2.5",
    page_icon="🎓",
    layout="wide"
)

# --- SIDEBAR: CONFIGURAÇÃO DA IA ---
with st.sidebar:
    st.header("⚙️ Configuração IA")
    provedor = st.selectbox("Provedor", ["OpenAI (GPT)", "Google (Gemini)"])
    st.session_state.api_key = st.text_input("Chave de API", type="password")
    if provedor.startswith("OpenAI"):
        modelo = st.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        modelo = st.selectbox("Modelo Gemini", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"])
    st.info("Gere questões ENADE a partir de texto-base ou deixe a IA gerar automaticamente.")

if not st.session_state.get("api_key"):
    st.warning("Informe a chave de API na lateral para continuar.")
    st.stop()

# --- ESTADO INICIAL ---
st.session_state.setdefault("text_base", "")
st.session_state.setdefault("auto", False)
st.session_state.setdefault("ref_final", "")
st.session_state.setdefault("questoes", [])
st.session_state.setdefault("search_results", [])

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600)
def extrair_conteudo_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string if soup.title else ""
        author_meta = soup.find("meta", attrs={"name": "author"})
        author = author_meta["content"] if author_meta else ""
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text, title, author
    except:
        return None, None, None

def extrair_texto_upload(upload):
    try:
        if upload.type == "application/pdf":
            reader = PyPDF2.PdfReader(BytesIO(upload.read()))
            return "".join(p.extract_text() or "" for p in reader.pages)
        else:
            doc = Document(BytesIO(upload.read()))
            return "\n".join(p.text for p in doc.paragraphs)
    except:
        return None

def chamar_llm(prompts, prov, mdl, temperature=0.7, max_tokens=2000):
    if prov.startswith("OpenAI"):
        client = OpenAI(api_key=st.session_state.api_key)
        r = client.chat.completions.create(
            model=mdl,
            messages=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "text"}
        )
        return r.choices[0].message.content.strip()
    else:
        genai.configure(api_key=st.session_state.api_key)
        cfg = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain"
        )
        m = genai.GenerativeModel(mdl)
        prompt_text = "\n".join(f"**{m0['role']}**: {m0['content']}" for m0 in prompts)
        resp = m.generate_content(prompt_text, generation_config=cfg)
        return resp.text

def search_articles(query, num=5):
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"}
    params = {"q": query, "hl": "pt-BR", "gl": "br", "num": num}
    r = requests.get("https://www.google.com/search", headers=headers, params=params, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for block in soup.select("div.yuRUbf")[:num]:
        a = block.find("a", href=True)
        h3 = block.find("h3")
        if a and h3:
            results.append({"title": h3.get_text(), "url": a["href"]})
    if not results:
        for g in soup.select("div.g")[:num]:
            a = g.find("a", href=True)
            h3 = g.find("h3")
            if a and h3:
                results.append({"title": h3.get_text(), "url": a["href"]})
    return results

# --- 1. Definição do Escopo ---
st.title("🎓 Gerador de Questões ENADE")
with st.container():
    st.header("1. Definição do Escopo")
    c1, c2, c3 = st.columns(3)
    area = c1.selectbox("Área", list(AREAS_ENADE.keys()))
    curso = c2.selectbox("Curso", AREAS_ENADE[area])
    assunto = c3.text_input("Assunto central", "")
    st.session_state.escopo = {"area": area, "curso": curso, "assunto": assunto}

# --- 1.1 Tipo de Questão ---
with st.container():
    st.header("1.1 Tipo de Questão")
    tipos_questao = {
        "Múltipla Escolha Tradicional": "apresentar enunciado + alternativas (1 correta)",
        "Múltiplas Respostas":          "enunciado + alternativas (mais de uma correta)",
        "Complementação":               "frase com lacuna '___', alternativas completam",
        "Afirmação-Razão":              "afirmação e razão, avaliar verdade e justificativa",
        "Resposta Múltipla":            "selecionar/agrupar várias corretas"
    }
    st.session_state.setdefault("question_type", list(tipos_questao.keys())[0])
    question_type = st.selectbox(
        "Selecione o tipo de questão",
        options=list(tipos_questao.keys()),
        format_func=lambda k: f"{k}: {tipos_questao[k]}",
        key="question_type"
    )

# --- 2. Texto-Base (Opcional) ---
with st.container():
    st.header("2. Texto-Base (Opcional)")
    opc = st.radio(
        "Deseja inserir um texto-base?",
        ["Não, IA gera automaticamente", "Sim, inserir texto-base"],
        horizontal=True
    )

    if opc.startswith("Não"):
        # sempre permitir nova contextualização
        if st.button("Gerar contextualização"):
            with st.spinner("Gerando contextualização..."):
                prompts = [
                    {
                        "role": "system",
                        "content": (
                            f"Você é um docente do {curso} que produz textos-base "
                            "contextualizados para questões do ENADE. Esses textos devem "
                            "possuir complexidade e utilizar conceitos e definições da área."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Gere um texto com no mínimo 5 frases para situação-"
                            f"problema da questão ENADE em Área: {area}, Curso: {curso}, "
                            f"Assunto: {assunto}. Não inclua nenhum comentário, apenas "
                            "o texto-base como saída."
                        )
                    }
                ]
                tb = chamar_llm(prompts, provedor, modelo, temperature=0.5, max_tokens=300)
                st.session_state.text_base = tb or ""
                st.session_state.auto = True
        if st.session_state.text_base:
            st.success("Contextualização gerada!")
    else:
        st.session_state.auto = False
        modo = st.radio(
            "Como fornecer o texto-base?",
            ["Upload de PDF", "Buscar artigos na internet"],
            horizontal=True
        )
        if modo == "Upload de PDF":
            up = st.file_uploader("Envie um PDF", type="pdf")
            if up:
                with st.spinner("Resumindo PDF..."):
                    txt = extrair_texto_upload(up)
                    if txt:
                        prompts = [
                            {"role": "system", "content": "Você gera resumos concisos para ENADE."},
                            {"role": "user",   "content": f"Resuma em até 7 frases para situação-problema:\n\n{txt}"}
                        ]
                        st.session_state.text_base = chamar_llm(prompts, provedor, modelo, temperature=0.4, max_tokens=250)
                        st.success("Resumo pronto!")
        else:
            try:
                if st.button("🔍 Buscar artigos", key="search_btn"):
                    st.session_state.search_results = []
                    with st.spinner("Buscando artigos..."):
                        res = search_articles(assunto, num=5)
                        if res:
                            st.session_state.search_results = res
                        else:
                            st.warning("Nenhum resultado encontrado.")
                if st.session_state.search_results:
                    opts = [f"{r['title']} — {r['url']}" for r in st.session_state.search_results]
                    sel = st.selectbox("Selecione um artigo", opts, key="sel_artigo")
                    if st.button("▶️ Usar artigo", key="use_btn"):
                        art = st.session_state.search_results[opts.index(sel)]
                        cont, tit, aut = extrair_conteudo_url(art["url"])
                        if cont:
                            with st.spinner("Extraindo e resumindo..."):
                                prompts = [
                                    {"role": "system", "content": "Você gera resumos concisos para ENADE."},
                                    {"role": "user",   "content": f"Resuma em até 6 frases para situação-problema:\n\n{cont}"}
                                ]
                                st.session_state.text_base = chamar_llm(prompts, provedor, modelo, temperature=0.4, max_tokens=250)
                                st.session_state.fonte_info = {
                                    "titulo": tit, "autor": aut,
                                    "veiculo": art["url"].split("/")[2], "link": art["url"]
                                }
                                st.success("Resumo pronto!")
            except Exception as e:
                st.error(f"Erro ao buscar artigos: {e}")

# --- 3. Texto-Base e Referência ---
if st.session_state.text_base:
    st.header("3. Texto-Base e Referência")
    st.session_state.text_base = st.text_area(
        "Texto-Base (edite se necessário):",
        st.session_state.text_base,
        height=200,
        key="text_base_editavel"
    )
    if not st.session_state.auto:
        info = st.session_state.get("fonte_info", {})
        c1, c2, c3, c4 = st.columns(4)
        autor    = c1.text_input("Autor (SOBRENOME, Nome)", value=info.get("autor",""))
        titulo   = c2.text_input("Título", value=info.get("titulo",""))
        veiculo  = c3.text_input("Veículo", value=info.get("veiculo",""))
        data_pub = c4.text_input("Data de publicação", placeholder="dd mmm. aaaa")
        if autor and titulo and veiculo and data_pub:
            hoje = datetime.now()
            meses = ["jan.","fev.","mar.","abr.","mai.","jun.","jul.","ago.","set.","out.","nov.","dez."]
            acesso = f"{hoje.day} {meses[hoje.month-1]}. {hoje.year}"
            st.session_state.ref_final = (
                f"{autor}. {titulo}. {veiculo}, {data_pub}. "
                f"Disponível em: <{info.get('link','N/D')}>. Acesso em: {acesso}."
            )
            st.text_area("Referência ABNT:", st.session_state.ref_final, height=80, key="ref_display")

# --- 4. Parâmetros e Geração da Questão ---
if st.session_state.text_base and (st.session_state.auto or st.session_state.ref_final):
    st.header("4. Parâmetros e Geração")
    with st.form("frm"):
        perfil      = st.text_input("Perfil do egresso")
        comp        = st.text_input("Competência")
        niv         = st.select_slider("Nível Bloom", options=BLOOM_LEVELS, value="Analisar")
        dificuldade = st.slider("Nível de dificuldade", 1, 5, 3, help="1=Muito fácil;5=Muito difícil")
        verbs       = st.multiselect("Verbos de comando", BLOOM_VERBS[niv], default=BLOOM_VERBS[niv][:2])
        obs         = st.text_area("Observações (opcional)")
        gerar       = st.form_submit_button("🚀 Gerar Questão")

    if gerar:
        with st.spinner("Gerando…"):
            qt = st.session_state.question_type
            sys_p = """
Você é docente especialista em produzir questão no estilo ENADE.
- Enunciado claro, usando texto-base, linguagem impessoal.
- Alternativas e gabarito conforme tipo de questão.
- Citações no padrão ABNT.
Saída em texto puro no formato: Contextualização, Enunciado, Alternativas, Gabarito e Justificativas.
"""
            # instruções detalhadas para cada tipo
            if qt == "Múltipla Escolha Tradicional":
                sys_p += "\n• Tipo Múltipla Escolha Tradicional: enunciado seguido de 5 alternativas, apenas 1 correta e 4 distratores plausíveis."
            elif qt == "Múltiplas Respostas":
                sys_p += "\n• Tipo Múltiplas Respostas: enunciado com 5 alternativas, mais de uma correta; no Gabarito liste todas separadas por vírgula (ex.: A,C)."
            elif qt == "Complementação":
                sys_p += "\n• Tipo Complementação: use '___' para lacuna no enunciado; alternativas completam corretamente a frase."
            elif qt == "Afirmação-Razão":
                sys_p += "\n• Tipo Afirmação-Razão: apresente afirmação e razão; o aluno julga se cada uma é verdadeira e se a razão justifica a afirmação; no Gabarito use: 'A verdadeira, R verdadeira e justifica', etc."
            elif qt == "Resposta Múltipla":
                sys_p += "\n• Tipo Resposta Múltipla: apresente várias alternativas que podem ser agrupadas ou selecionadas múltiplas como corretas; indique no Gabarito todas as relações ou corretas."

            ref_txt = "" if st.session_state.auto else f"\nREFERÊNCIA:\n{st.session_state.ref_final}\n"
            usr_p = f"""
Área: {area}
Curso: {curso}
Assunto: {assunto}
Perfil: {perfil}
Competência: {comp}
Tipo de questão: {qt}
Dificuldade: {dificuldade}/5
Verbos: {', '.join(verbs)}
Observações: {obs}

TEXTO-BASE:
{st.session_state.text_base}
{ref_txt}

Por favor, siga EXATAMENTE o formato e não altere o texto-base.
"""
            out = chamar_llm(
                [{"role": "system", "content": sys_p},
                 {"role": "user",   "content": usr_p}],
                provedor, modelo,
                temperature=0.5, max_tokens=1000
            )
            if out:
                st.session_state.questoes.append(out)
                st.success("Questão gerada!")

# --- 5. Resultados, Download & Nova Questão ---
if st.session_state.questoes:
    st.warning("⚠️ Revise antes de usar.")
    st.header("5. Questões Geradas")
    for i, q in enumerate(st.session_state.questoes, 1):
        st.markdown(f"---\n**Questão #{i}**\n```\n{q}\n```")

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "📄 Baixar última (.txt)",
        "\n\n".join(st.session_state.questoes[-1:]),
        f"questao_{len(st.session_state.questoes)}.txt",
        "text/plain"
    )
    df_all = pd.DataFrame({"questão": st.session_state.questoes})
    to_xl = BytesIO()
    df_all.to_excel(to_xl, index=False, sheet_name="Questões")
    to_xl.seek(0)
    c2.download_button(
        "📥 Baixar todas (.xlsx)",
        to_xl,
        "todas_questoes_enade.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    if c3.button("💾 Salvar banco e Nova Questão"):
        df_all.to_excel("banco_questoes.xlsx", index=False, sheet_name="Questões")
        st.success("Banco salvo em banco_questoes.xlsx")
        st.session_state.text_base = ""
        st.session_state.auto = False
        st.session_state.ref_final = ""
        st.session_state.questoes = []
        st.session_state.search_results = []
        st.experimental_rerun()
