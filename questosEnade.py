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
for key in ("texto_fonte", "trecho_para_prompt", "contexto",
            "questao_bruta", "questao", "last_pdf"):
    if key not in st.session_state:
        st.session_state[key] = "" if key not in ("questao", "last_pdf") else None
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

# --- EXTRAÇÃO DE TEXTO ---
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
        st.error(f"Erro ao extrair URL: {e}")
        return None

@st.cache_data
def extrair_texto_pdf(upload) -> str | None:
    try:
        reader = PyPDF2.PdfReader(BytesIO(upload.read()))
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Erro ao ler PDF: {e}")
        return None

# --- RESUMO DO TEXTO-BASE ---
def gerar_resumo_llm(texto: str, api_key: str, modelo: str) -> str:
    prompt = f"""
Resuma em até 3 frases este texto, mantendo foco nos conceitos fundamentais, 
para servir de base a uma situação‐problema ENADE:

\"\"\"{texto}\"\"\"
"""
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": "Você é um assistente que cria resumos concisos para questões ENADE."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.5,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# --- GERAÇÃO DO CONTEXTO ---
def gerar_contexto_llm(texto_base: str, api_key: str, modelo: str) -> str:
    prompt = f"""
Com base neste trecho de texto-base, gere UMA BREVE situação-problema (contexto) 
profissional e relevante para uma questão ENADE. Retorne apenas o texto do contexto.

\"\"\"{texto_base}\"\"\"
"""
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": "Você é um assistente que elabora contextos para questões ENADE."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# --- PROMPT E GERAÇÃO DA QUESTÃO ---
SYSTEM_PROMPT = """
Você é um docente especialista no ENADE (INEP). Siga este checklist:
1. Use o contexto fornecido.
2. Apresente um texto-base referenciado (Autor/Veículo, Ano, Link/Arquivo).
3. Elabore um enunciado afirmativo, claro e objetivo.
4. Gere exatamente 5 alternativas (A–E), só 1 correta.
5. Distratores plausíveis, baseados em erros comuns.
6. Use linguagem formal, impessoal, norma-padrão.
7. Avalie aplicação de conhecimento (competência), não memorização.
8. Evite termos absolutos (“sempre”, “nunca”, etc.).
9. Indique gabarito no formato: "Gabarito: Letra X".
10. Inclua justificativas breves para cada alternativa.

Formato de saída (JSON):
{
  "contexto": "...",
  "texto_base": "...",
  "referencia": "...",
  "enunciado": "...",
  "alternativas": { "A":"", "B":"", "C":"", "D":"", "E":"" },
  "gabarito": "Letra X",
  "justificativas": { "A":"", "B":"", "C":"", "D":"", "E":"" }
}
"""

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

# --- SIDEBAR: API KEY E MODELO ---
with st.sidebar:
    st.markdown("## 🔑 Configuração da API\n- **OpenAI**: platform.openai.com/account/api-keys")
    api_key = st.text_input("Chave da OpenAI", type="password")
    modelo  = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-3.5-turbo"])
    if not api_key:
        st.warning("Insira sua chave de API para continuar.")
        st.stop()

# --- ETAPA 1: ESCOPO ---
st.header("1. Definição do Escopo")
area    = st.selectbox("Grande Área", list(AREAS_ENADE.keys()))
curso   = st.selectbox("Curso", AREAS_ENADE[area])
assunto = st.text_input("Tópico/Assunto central")

# --- ETAPA 2: TEXTO-BASE E TRECHO-BASE ---
st.header("2. Texto-Base e Trecho-Base")
col1, col2 = st.columns(2)
with col1:
    url = st.text_input("URL do artigo:", value=st.session_state.fonte_info["link"])
    if st.button("Extrair texto da URL"):
        txt = extrair_texto_url(url)
        if txt:
            st.session_state.texto_fonte = txt
            st.session_state.fonte_info["link"] = url
with col2:
    up = st.file_uploader("Ou envie um PDF", type=["pdf"])
    if up and up != st.session_state.last_pdf:
        txt = extrair_texto_pdf(up)
        if txt:
            st.session_state.texto_fonte = txt
            st.session_state.fonte_info["link"] = up.name
            st.session_state.last_pdf = up

if st.session_state.texto_fonte:
    st.success("Texto-base carregado!")
    with st.expander("Ver texto completo"):
        st.text_area("Texto-Fonte", st.session_state.texto_fonte, height=300)

    metodo_tb = st.radio(
        "Como obter o trecho-base?",
        ["Selecionar manualmente", "Gerar resumo automático com IA"]
    )
    if metodo_tb == "Selecionar manualmente":
        pars = [p.strip() for p in st.session_state.texto_fonte.split("\n") if len(p.strip()) > 80]
        sel = st.multiselect(
            "Selecione parágrafos",
            options=pars,
            format_func=lambda p: textwrap.shorten(p, 120, placeholder="…")
        )
        if sel:
            st.session_state.trecho_para_prompt = "\n\n".join(sel)
    else:
        if st.button("▶️ Gerar Resumo do Documento"):
            resumo = gerar_resumo_llm(st.session_state.texto_fonte, api_key, modelo)
            st.session_state.trecho_para_prompt = resumo
        if st.session_state.trecho_para_prompt:
            st.markdown("**Resumo gerado (edite se quiser):**")
            st.session_state.trecho_para_prompt = st.text_area(
                "Resumo para trecho-base",
                value=st.session_state.trecho_para_prompt,
                height=120
            )

# --- ETAPA 3: GERAÇÃO E EDIÇÃO DO CONTEXTO ---
if st.session_state.trecho_para_prompt and not st.session_state.contexto:
    with st.spinner("Gerando contexto..."):
        st.session_state.contexto = gerar_contexto_llm(
            st.session_state.trecho_para_prompt, api_key, modelo
        )

if st.session_state.contexto:
    st.header("3. Contexto (situação-problema)")
    st.session_state.contexto = st.text_area(
        "Edite o contexto conforme desejar:",
        value=st.session_state.contexto,
        height=120
    )

# --- ETAPA 4: PARÂMETROS E GERAÇÃO DA QUESTÃO ---
if st.session_state.contexto:
    st.header("4. Parâmetros ENADE e Geração")
    with st.form("enade_form"):
        autor      = st.text_input("Autor (SOBRENOME, Nome)")
        titulo     = st.text_input("Título do texto-base")
        fonte      = st.text_input("Veículo (revista, jornal, site etc.)")
        data_pub   = st.text_input("Data de publicação (dia mês abreviado. ano)")
        tipo_item  = st.selectbox("Tipo de item", ["Múltipla Escolha", "Asserção-Razão", "Discursivo"])
        perfil     = st.text_input("Perfil do egresso")
        competencia= st.text_input("Competência a ser avaliada")
        objeto     = st.text_input("Objeto de conhecimento")
        dificuldade= st.select_slider("Nível de dificuldade", ["Fácil", "Média", "Difícil"])
        extra      = st.text_area("Observações (opcional)")
        gerar_btn  = st.form_submit_button("🚀 Gerar Questão")
    if gerar_btn:
        if not (autor and titulo and fonte and data_pub):
            st.error("Preencha Autor, Título, Veículo e Data de publicação.")
        else:
            hoje = datetime.now()
            meses_abnt = ["jan.", "fev.", "mar.", "abr.",
                          "mai.", "jun.", "jul.", "ago.",
                          "set.", "out.", "nov.", "dez."]
            acesso = f"{hoje.day} {meses_abnt[hoje.month-1]} {hoje.year}"
            referencia_abnt = (
                f"{autor}. {titulo}. {fonte}, {data_pub}. "
                f"Disponível em: <{st.session_state.fonte_info['link']}>. "
                f"Acesso em: {acesso}."
            )
            prompt = f"""
**Contexto (situação-problema):**
{st.session_state.contexto}

**Texto-Base:**
{st.session_state.trecho_para_prompt}

**Referência (ABNT):**
{referencia_abnt}

**Encomenda ENADE:**
- Curso: {curso}
- Assunto: {assunto}
- Tipo de item: {tipo_item}
- Perfil do egresso: {perfil}
- Competência: {competencia}
- Objeto de conhecimento: {objeto}
- Nível de dificuldade: {dificuldade}
- Observações: {extra}
"""
            raw = gerar_questao_llm(prompt, api_key, modelo)
            st.session_state.questao_bruta = raw
            try:
                q = json.loads(raw)
                campos = {"contexto", "texto_base", "referencia",
                          "enunciado", "alternativas", "gabarito", "justificativas"}
                faltando = campos - set(q.keys())
                if faltando:
                    st.error(f"Faltam campos na resposta: {faltando}")
                else:
                    st.session_state.questao = q
            except Exception as e:
                st.error(f"Resposta não é JSON válido: {e}")

# --- ETAPA 5: EXIBIÇÃO FINAL ---
if st.session_state.questao:
    st.header("5. Questão ENADE Estruturada")
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
        "📥 Baixar (.json)",
        data=json.dumps(q, ensure_ascii=False, indent=2),
        file_name=f"questao_{curso.replace(' ', '_')}.json",
        mime="application/json"
    )
else:
    st.info("Siga todos os passos para gerar sua questão ENADE.")
