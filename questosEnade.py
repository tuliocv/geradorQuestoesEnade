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
    page_title="Gerador de Questões ENADE v2.6",
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
st.session_state.setdefault("perfil", "")
st.session_state.setdefault("competencia", "")


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
        else: # Assumindo docx
            doc = Document(BytesIO(upload.read()))
            return "\n".join(p.text for p in doc.paragraphs)
    except:
        return None

def chamar_llm(prompts, prov, mdl, temperature=0.7, max_tokens=2000):
    try:
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
    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
        return None

# NOVA FUNÇÃO DE BUSCA COM OPÇÃO PARA NOTÍCIAS
def search_articles(query, num=5, search_type='web'):
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"}
    params = {"q": query, "hl": "pt-BR", "gl": "br", "num": num}
    if search_type == 'news':
        params['tbm'] = 'nws' # Parâmetro para buscar em "Notícias"

    try:
        r = requests.get("https://www.google.com/search", headers=headers, params=params, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        
        # Seletores comuns para resultados de busca e notícias
        for block in soup.select("div.SoaBEf, div.yuRUbf"):
            a = block.find("a", href=True)
            title_element = block.find("h3") or block.find('div', role='heading')
            if a and title_element:
                 results.append({"title": title_element.get_text(), "url": a["href"]})
            if len(results) >= num:
                break
        
        return results
    except Exception as e:
        st.error(f"Erro na busca: {e}")
        return []

# --- 1. Definição do Escopo ---
st.title("🎓 Gerador de Questões ENADE")
st.header("1. Definição do Escopo")
c1, c2, c3 = st.columns(3)
area = c1.selectbox("Área", list(AREAS_ENADE.keys()))
curso = c2.selectbox("Curso", AREAS_ENADE[area])
assunto = c3.text_input("Assunto central", "")
st.session_state.escopo = {"area": area, "curso": curso, "assunto": assunto}

# --- 1.1 Tipo de Questão ---
st.header("1.1 Tipo de Questão")
tipos_questao = {
    "Múltipla Escolha Tradicional": "apresentar enunciado + alternativas (1 correta)",
    "Complementação": "frase com lacuna '___', alternativas completam",
    "Afirmação-Razão": "afirmação e razão, avaliar verdade e justificativa",
    "Resposta Múltipla": "selecionar/agrupar várias corretas"
}
st.session_state.setdefault("question_type", list(tipos_questao.keys())[0])
question_type = st.selectbox(
    "Selecione o tipo de questão",
    options=list(tipos_questao.keys()),
    format_func=lambda k: f"{k}: {tipos_questao[k]}",
    key="question_type"
)


# --- 2. Definição Pedagógica e Geração do Texto-Base ---
st.header("2. Definição Pedagógica e Geração do Texto-Base")

# INPUTS DE PERFIL E COMPETÊNCIA AGORA SÃO FEITOS ANTES
st.session_state.perfil = st.text_input(
    "Perfil do egresso a ser avaliado", 
    st.session_state.perfil,
    help="Descreva o que se espera do profissional formado."
)
st.session_state.competencia = st.text_input(
    "Competência a ser avaliada", 
    st.session_state.competencia,
    help="Descreva a habilidade ou conhecimento específico que a questão deve medir."
)

opc = st.radio(
    "Selecione a fonte do texto-base:",
    ["Gerar texto com IA (Recomendado)", "Fornecer um texto-base (Upload ou Busca)"],
    horizontal=True
)

if opc.startswith("Gerar"):
    st.session_state.auto = True
    if st.session_state.perfil and st.session_state.competencia:
        if st.button("Gerar contextualização com IA"):
            with st.spinner("Gerando contextualização alinhada ao perfil e competência..."):
                prompts = [
                    {
                        "role": "system",
                        "content": (
                            f"Você é um docente especialista do curso de {curso} que cria textos-base para questões do ENADE. "
                            "Os textos devem ser contextualizados, apresentar uma situação-problema e ser perfeitamente alinhados "
                            "às diretrizes pedagógicas fornecidas."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Elabore um texto-base (entre 150 e 250 palavras) para uma questão do ENADE. "
                            f"O texto deve ser uma situação-problema sobre o assunto '{assunto}', "
                            f"voltada para um egresso com o seguinte perfil: '{st.session_state.perfil}'. "
                            f"A avaliação focará na seguinte competência: '{st.session_state.competencia}'. "
                            "O texto deve ser complexo, rico em conceitos da área e estritamente técnico. "
                            "Não inclua o enunciado da questão nem alternativas, apenas o texto-base."
                        )
                    }
                ]
                tb = chamar_llm(prompts, provedor, modelo, temperature=0.6, max_tokens=400)
                if tb:
                    st.session_state.text_base = tb
                    st.success("Contextualização gerada!")
    else:
        st.warning("Preencha o Perfil do egresso e a Competência para gerar o texto-base com a IA.")

else: # Fornecer texto-base
    st.session_state.auto = False
    modo = st.radio(
        "Como deseja fornecer o texto-base?",
        ["Upload de PDF", "Buscar na internet"],
        horizontal=True
    )
    if modo == "Upload de PDF":
        up = st.file_uploader("Envie um PDF", type=["pdf", "docx"])
        if up:
            with st.spinner("Extraindo e resumindo o documento..."):
                txt = extrair_texto_upload(up)
                if txt:
                    prompts = [
                        {"role": "system", "content": "Você é um especialista em resumir textos acadêmicos para serem usados como base em questões do ENADE."},
                        {"role": "user", "content": f"Resuma o texto a seguir em um parágrafo de até 200 palavras, focando em criar uma situação-problema relevante para o curso de {curso} sobre o tema {assunto}.\n\nTexto Original:\n{txt}"}
                    ]
                    st.session_state.text_base = chamar_llm(prompts, provedor, modelo, temperature=0.4, max_tokens=300)
                    st.success("Resumo pronto!")
                else:
                    st.error("Não foi possível extrair texto do arquivo.")
    
    else: # Buscar na internet
        c1_search, c2_search = st.columns(2)
        
        # BOTÕES SEPARADOS PARA BUSCA ACADÊMICA E DE NOTÍCIAS
        if c1_search.button("🔍 Buscar artigos acadêmicos", key="search_acad_btn"):
            st.session_state.search_results = []
            with st.spinner(f"Buscando artigos acadêmicos sobre '{assunto}'..."):
                res = search_articles(f"{assunto} filetype:pdf site:.edu OR site:.org", num=5, search_type='web')
                st.session_state.search_results = res if res else []
                if not res: st.warning("Nenhum resultado encontrado.")

        if c2_search.button("📰 Buscar notícias", key="search_news_btn"):
            st.session_state.search_results = []
            with st.spinner(f"Buscando notícias sobre '{assunto}'..."):
                res = search_articles(f'"{assunto}"', num=5, search_type='news')
                st.session_state.search_results = res if res else []
                if not res: st.warning("Nenhuma notícia encontrada.")

        if st.session_state.search_results:
            opts = [f"{r['title']} — {r['url']}" for r in st.session_state.search_results]
            sel = st.selectbox("Selecione um artigo ou notícia", opts, key="sel_artigo")
            if st.button("▶️ Usar este conteúdo", key="use_btn"):
                art = st.session_state.search_results[opts.index(sel)]
                cont, tit, aut = extrair_conteudo_url(art["url"])
                if cont:
                    with st.spinner("Extraindo e resumindo..."):
                        prompts = [
                            {"role": "system", "content": "Você resume conteúdos da web para serem usados como base em questões de prova do ENADE."},
                            {"role": "user", "content": f"Resuma o conteúdo a seguir em um parágrafo conciso (até 200 palavras) que sirva como situação-problema para uma questão sobre '{assunto}'.\n\nConteúdo:\n{cont}"}
                        ]
                        st.session_state.text_base = chamar_llm(prompts, provedor, modelo, temperature=0.4, max_tokens=300)
                        st.session_state.fonte_info = {
                            "titulo": tit, "autor": aut,
                            "veiculo": art["url"].split("/")[2], "link": art["url"]
                        }
                        st.success("Resumo pronto!")
                else:
                    st.error(f"Não foi possível extrair conteúdo da URL: {art['url']}")

# --- 3. Edição do Texto-Base e Geração da Questão ---
if st.session_state.text_base:
    st.header("3. Parâmetros e Geração da Questão")
    st.session_state.text_base = st.text_area(
        "Texto-Base (edite se necessário):",
        st.session_state.text_base,
        height=200,
        key="text_base_editavel"
    )

    ref_placeholder = "dd mmm. aaaa (ex: 10 jul. 2025)"
    if not st.session_state.auto: # Se não foi auto-gerado, pede referência
        info = st.session_state.get("fonte_info", {})
        st.subheader("Referência ABNT")
        c1, c2, c3, c4 = st.columns(4)
        autor = c1.text_input("Autor (SOBRENOME, Nome)", value=info.get("autor",""))
        titulo = c2.text_input("Título", value=info.get("titulo",""))
        veiculo = c3.text_input("Veículo", value=info.get("veiculo",""))
        data_pub = c4.text_input("Data de publicação", placeholder=ref_placeholder)
        
        if all([autor, titulo, veiculo, data_pub]):
            hoje = datetime.now()
            meses = ["jan.","fev.","mar.","abr.","mai.","jun.","jul.","ago.","set.","out.","nov.","dez."]
            acesso = f"{hoje.day} {meses[hoje.month-1]}. {hoje.year}"
            st.session_state.ref_final = (
                f"{autor}. **{titulo}**. {veiculo}, {data_pub}. "
                f"Disponível em: <{info.get('link','N/D')}>. Acesso em: {acesso}."
            )
            st.text_area("Referência Formatada:", st.session_state.ref_final, height=80, key="ref_display", disabled=True)

    if st.session_state.auto or st.session_state.ref_final:
        with st.form("frm"):
            st.subheader("Parâmetros Finais da Questão")
            niv = st.select_slider("Nível Bloom", options=BLOOM_LEVELS, value="Analisar")
            dificuldade = st.slider("Nível de dificuldade", 1, 5, 3, help="1=Muito fácil; 5=Muito difícil")
            verbs = st.multiselect("Verbos de comando", BLOOM_VERBS[niv], default=BLOOM_VERBS[niv][:2])
            obs = st.text_area("Observações/Instruções adicionais para a IA (opcional)")
            gerar = st.form_submit_button("🚀 Gerar Questão")

            if gerar:
                with st.spinner("A IA está gerando a questão completa. Aguarde..."):
                    sys_p = """
Você é um docente especialista em produzir questões no estilo ENADE, seguindo rigorosamente as diretrizes.
- A questão deve ser inédita e alinhada à encomenda (perfil, competência, conteúdo).
- O texto-base fornecido deve ser usado como ponto de partida para a contextualização. O enunciado deve ser claro e afirmativo.
- É proibido solicitar a alternativa "incorreta" ou "exceto".
- Múltipla Escolha: Apenas 1 alternativa correta e 4 distratores plausíveis, que explorem erros conceituais comuns.
- Discursivas: A tarefa deve ser complexa (análise, argumentação) e o padrão de resposta deve ser detalhado.
- Afirmação-Razão: Avalie a veracidade de duas asserções e a relação de causalidade entre elas.
- Resposta Múltipla: Apresente 3 a 5 afirmativas numeradas (I, II, III...) e as alternativas devem combinar as corretas.
- A saída deve ser um texto puro, seguindo o formato especificado, incluindo justificativas detalhadas para cada alternativa.
"""
                    ref_txt = "" if st.session_state.auto else f"\nREFERÊNCIA DO TEXTO-BASE:\n{st.session_state.ref_final}\n"
                    usr_p = f"""
GERAR QUESTÃO ENADE:

**1. DIRETRIZES PEDAGÓGICAS:**
   - **Área:** {area}
   - **Curso:** {curso}
   - **Assunto:** {assunto}
   - **Perfil do Egresso:** {st.session_state.perfil}
   - **Competência Avaliada:** {st.session_state.competencia}

**2. PARÂMETROS DA QUESTÃO:**
   - **Tipo de Questão:** {question_type}
   - **Nível de Dificuldade:** {dificuldade}/5
   - **Nível de Bloom (Verbos de Comando):** {niv} ({', '.join(verbs)})
   - **Observações Adicionais:** {obs if obs else "Nenhuma."}

**3. TEXTO-BASE (OBRIGATÓRIO):**
{st.session_state.text_base}
{ref_txt}

**4. FORMATO DE SAÍDA (OBRIGATÓRIO):**
Use EXATAMENTE este formato de saída em texto puro, sem comentários adicionais:

ENUNCIADO:
[Crie aqui o enunciado da questão, conectando o texto-base com o comando da questão. Use os verbos de comando.]

ALTERNATIVAS:
A. [Alternativa A]
B. [Alternativa B]
C. [Alternativa C]
D. [Alternativa D]
E. [Alternativa E]

GABARITO:
Letra X

JUSTIFICATIVAS:
A. [Justificativa detalhada explicando por que a alternativa A está incorreta ou correta.]
B. [Justificativa detalhada explicando por que a alternativa B está incorreta ou correta.]
C. [Justificativa detalhada explicando por que a alternativa C está incorreta ou correta.]
D. [Justificativa detalhada explicando por que a alternativa D está incorreta ou correta.]
E. [Justificativa detalhada explicando por que a alternativa E está incorreta ou correta.]
"""
                    out = chamar_llm(
                        [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                        provedor, modelo,
                        temperature=0.5, max_tokens=1500
                    )
                    if out:
                        # Adiciona o texto-base e a referência no topo da questão gerada para o display
                        full_question_output = f"TEXTO-BASE:\n{st.session_state.text_base}\n\n{ref_txt if not st.session_state.auto else ''}{out}"
                        st.session_state.questoes.append(full_question_output)
                        st.success("Questão gerada com sucesso!")

# --- 4. Resultados, Download & Nova Questão ---
if st.session_state.questoes:
    st.header("4. Questões Geradas")
    st.warning("⚠️ Lembre-se de revisar cuidadosamente o conteúdo gerado pela IA antes de utilizá-lo oficialmente.")
    
    for i, q in enumerate(st.session_state.questoes, 1):
        st.markdown(f"---")
        st.markdown(f"#### Questão #{i}")
        st.text_area(f"Questão Gerada #{i}", q, height=400, key=f"q_area_{i}")

    c1, c2, c3 = st.columns(3)
    
    # Prepara os dados para download
    last_question_text = st.session_state.questoes[-1]
    all_questions_text = "\n\n---\n\n".join(st.session_state.questoes)
    
    df_all = pd.DataFrame({"questão": st.session_state.questoes})
    to_xl = BytesIO()
    df_all.to_excel(to_xl, index=False, sheet_name="Questões")
    to_xl.seek(0)

    c1.download_button(
        "📄 Baixar última (.txt)",
        last_question_text,
        f"enade_questao_{len(st.session_state.questoes)}.txt",
        "text/plain"
    )
    
    c2.download_button(
        "📥 Baixar todas (.xlsx)",
        to_xl,
        "banco_questoes_enade.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if c3.button("✨ Gerar Nova Questão (Limpar Tudo)"):
        # Limpa o estado para começar de novo
        keys_to_reset = ["text_base", "auto", "ref_final", "questoes", "search_results", "perfil", "competencia"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
