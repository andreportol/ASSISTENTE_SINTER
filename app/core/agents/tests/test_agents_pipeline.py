import pytest

from app.core.agents.semantic_interpreter import interpret
from app.core.agents.conversation_state import ConversationState
from app.core.agents.orchestrator import (
    AnalyticsPipeline,
    UnsupportedQuestionError,
    _build_state,
    _select_executor,
)
from app.core.agents.sql_executor import (
    CountMovimentacoesPorDiaExecutor,
    ListProdutosPorDiaExecutor,
    CountVendasPorDiaExecutor,
    SumProdutosVendidosPorDiaExecutor,
    TopProdutosPorDiaExecutor,
    TopProdutosPorPeriodoExecutor,
    TotalVendasFinanceiroPorDiaExecutor,
)


# ============================================================
# 1️⃣ CONTAGEM DE SAÍDAS DE PRODUTOS (regra de domínio)
# ============================================================

def test_count_saidas_de_produtos():
    question = "Quantas saídas de produtos teve no dia 24/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "count"
    assert final_state["objeto"] == "movimentacao"  # regra de domínio
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-24"

    executor = _select_executor(final_state)
    assert isinstance(executor, CountMovimentacoesPorDiaExecutor)


# ============================================================
# 2️⃣ LISTAGEM DE PRODUTOS VENDIDOS
# ============================================================

def test_list_produtos_vendidos():
    question = "Quais os nomes dos produtos vendidos no dia 24/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "list"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-24"

    executor = _select_executor(final_state)
    assert isinstance(executor, ListProdutosPorDiaExecutor)


# ============================================================
# 3️⃣ TOTAL FINANCEIRO DE VENDAS
# ============================================================

def test_total_vendas_financeiro():
    question = "Qual foi o total de vendas em reais no dia 24/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "sum"
    assert final_state["objeto"] == "venda"
    assert final_state["data"] == "2025-12-24"

    executor = _select_executor(final_state)
    assert isinstance(executor, TotalVendasFinanceiroPorDiaExecutor)


# ============================================================
# 4️⃣ CONTAGEM DE VENDAS (sem produto explícito)
# ============================================================

def test_count_vendas():
    question = "Quantas vendas foram realizadas no dia 24/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "count"
    assert final_state["objeto"] == "venda"
    assert final_state["data"] == "2025-12-24"

    executor = _select_executor(final_state)
    assert isinstance(executor, CountVendasPorDiaExecutor)


# ============================================================
# 5️⃣ BLOQUEIO DE PERGUNTA SUBJETIVA
# ============================================================

def test_pergunta_subjetiva_bloqueada():
    question = "Qual foi o produto mais bonito vendido em dezembro?"

    semantic = interpret(question)

    assert semantic.get("unsupported") is True

    pipeline = AnalyticsPipeline()

    with pytest.raises(UnsupportedQuestionError):
        pipeline.run(question)


# ============================================================
# 6️⃣ TOP 1 PRODUTO MAIS VENDIDO
# ============================================================

def test_top_produto_mais_vendido():
    question = "Qual o produto mais vendido no dia 22/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-22"
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 1

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorDiaExecutor)


# ============================================================
# 7️⃣ TOP 5 PRODUTOS MAIS VENDIDOS
# ============================================================

def test_top_5_produtos_mais_vendidos():
    question = "Quais os 5 produtos mais vendidos em 22/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-22"
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 5

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorDiaExecutor)


# ============================================================
# 8️⃣ TOP 10 PRODUTOS POR FATURAMENTO
# ============================================================

def test_top_10_produtos_por_faturamento():
    question = "Top 10 produtos por faturamento em 22/12/2025"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-22"
    assert final_state["metric"] == "valor"
    assert final_state["rank_limit"] == 10

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorDiaExecutor)


# ============================================================
# 9️⃣ QUANTIDADE DE PRODUTOS VENDIDOS (SOMA)
# ============================================================

def test_sum_produtos_vendidos():
    question = "Quantos produtos foram vendidos no dia 22/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "sum"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-22"
    assert final_state["metric"] == "quantidade"

    executor = _select_executor(final_state)
    assert isinstance(executor, SumProdutosVendidosPorDiaExecutor)


# ============================================================
# 🔟 TOP SEM DATA EXPLICITA
# ============================================================

def test_top_produtos_sem_data():
    question = "Quais os produtos mais vendidos?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state.get("data") is None
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 1

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorDiaExecutor)


# ============================================================
# 1️⃣1️⃣ TOP COM "NO DIA" HERDA DATA DO CONTEXTO
# ============================================================

def test_top_produtos_no_dia_herda_data():
    history = [
        ("user", "Quantas vendas foram realizadas no dia 22/12/2025?"),
        ("assistant", "Foram realizadas 3047 vendas no dia 2025-12-21."),
        ("user", "Quais os produtos mais vendidos?"),
    ]
    question = "Quais foram os 10 produtos mais vendidos no dia."

    final_state = _build_state(history, interpret(question)).to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-22"
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 10
    assert final_state["date_anchor"] == "dia"

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorDiaExecutor)


# ============================================================
# 1️⃣2️⃣ SEGUNDO PRODUTO MAIS VENDIDO
# ============================================================

def test_segundo_produto_mais_vendido():
    question = "O segundo produto mais vendido no dia 24/12/2025?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-24"
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 2
    assert final_state["rank_position"] == 2

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorDiaExecutor)


# ============================================================
# 1️⃣3️⃣ "DEPOIS DO" IMPLICA SEGUNDO
# ============================================================

def test_depois_do_segundo_mais_vendido():
    question = (
        "Depois do rejunte cerâmico modelo 165 qual foi o produto mais vendido "
        "no dia 24/12/2025?"
    )

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["data"] == "2025-12-24"
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 2
    assert final_state["rank_position"] == 2

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorDiaExecutor)


# ============================================================
# 1️⃣4️⃣ TOP MENSAL
# ============================================================

def test_top_produto_mes():
    question = (
        "No mês de dezembro de 2025, qual foi o produto mais vendido "
        "e quantas unidades desse produto foram comercializadas?"
    )

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["date_start"] == "2025-12-01"
    assert final_state["date_end"] == "2025-12-31"
    assert final_state["periodo"] == "mes"
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 1

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorPeriodoExecutor)


# ============================================================
# 1️⃣5️⃣ TOP ANUAL
# ============================================================

def test_top_produto_ano():
    question = "Em 2025, qual foi o produto mais vendido por faturamento?"

    semantic = interpret(question)
    state = ConversationState()
    state.update(semantic)

    final_state = state.to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["date_start"] == "2025-01-01"
    assert final_state["date_end"] == "2025-12-31"
    assert final_state["periodo"] == "ano"
    assert final_state["metric"] == "valor"
    assert final_state["rank_limit"] == 1

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorPeriodoExecutor)


# ============================================================
# 1️⃣6️⃣ TOP MENSAL SEM ANO EXPLÍCITO (HERDA DO CONTEXTO)
# ============================================================

def test_top_produto_mes_sem_ano():
    history = [
        ("user", "No mês de dezembro de 2025, qual foi o produto mais vendido?"),
    ]
    question = "Eu quero saber o produto mais vendido no mês de novembro?"

    final_state = _build_state(history, interpret(question, history)).to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["date_start"] == "2025-11-01"
    assert final_state["date_end"] == "2025-11-30"
    assert final_state["periodo"] == "mes"
    assert final_state["metric"] == "quantidade"
    assert final_state["rank_limit"] == 1

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorPeriodoExecutor)


# ============================================================
# 1️⃣7️⃣ BLOQUEIO DE MES SEM ANO (SEM CONTEXTO)
# ============================================================

def test_mes_sem_ano_pede_ano():
    question = "Qual o produto mais vendido no mês de novembro?"

    semantic = interpret(question)

    assert semantic.get("needs_confirmation") is True
    assert semantic.get("confirmation") == "Qual ano?"

    pipeline = AnalyticsPipeline()
    assert pipeline.run(question) == "Qual ano?"


# ============================================================
# 1️⃣8️⃣ RESPOSTA COM ANO APOS CONFIRMACAO
# ============================================================

def test_mes_sem_ano_responde_com_ano():
    history = [
        ("user", "Qual o produto mais vendido no mês de novembro?"),
        ("assistant", "Qual ano?"),
    ]
    question = "2025"

    final_state = _build_state(history, interpret(question, history)).to_dict()

    assert final_state["intent"] == "top"
    assert final_state["objeto"] == "produto"
    assert final_state["tipo_movimento"] == "saida"
    assert final_state["date_start"] == "2025-11-01"
    assert final_state["date_end"] == "2025-11-30"
    assert final_state["periodo"] == "mes"

    executor = _select_executor(final_state)
    assert isinstance(executor, TopProdutosPorPeriodoExecutor)
