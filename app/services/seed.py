from __future__ import annotations

from app.models import EvalSample, ProjectContext
from app.storage import SQLiteStore


def ensure_seed_data(store: SQLiteStore) -> None:
    """首次启动写入一个示例项目，帮助用户完成端到端演示。"""
    if store.list_project_contexts():
        return
    context = ProjectContext(
        name="电商订单问答评测示例",
        project_background="这是一个面向电商订单、商品、退款数据的中文问答系统评测项目。用户会询问订单状态、退款金额、商品信息和统计口径。",
        system_description="被评测系统是一个黑盒 RAG / 数据问答 API，应该基于订单表、商品表、退款表和业务规则回答问题。",
        evaluation_goals="重点评估答案准确性、证据支持、不可回答时的拒答能力、业务规则遵循情况，以及延迟和 token 成本。",
        business_rules="""1. 查询退款金额时必须引用 refunds 表。
2. 用户问题没有时间范围时，应说明默认使用全部可见数据或要求补充时间范围。
3. 涉及订单状态判断时，必须基于 orders.status 字段。
4. 如果上下文没有足够字段支持结论，应拒答或说明无法判断。""",
        question_type_instructions="""事实核对
多跳推理
业务规则
边界条件
不可回答""",
        uploaded_assets=[
            {
                "file_name": "demo_schema.sql",
                "file_type": "sql",
                "size_bytes": 420,
                "excerpt": """orders(order_id, user_id, order_date, status, total_amount)
order_items(order_id, sku_id, quantity, unit_price)
products(sku_id, product_name, category, is_active)
refunds(refund_id, order_id, refund_amount, refund_reason, refund_date)""",
            },
            {
                "file_name": "demo_rows.csv",
                "file_type": "csv",
                "size_bytes": 260,
                "excerpt": """table,values
orders,"O1001,U01,2026-03-01,paid,299.00"
order_items,"O1001,SKU88,1,299.00"
products,"SKU88,智能手环,数码,true"
refunds,"R9001,O1002,99.00,质量问题,2026-03-03" """,
            },
        ],
        schema_text="""orders(order_id, user_id, order_date, status, total_amount)
order_items(order_id, sku_id, quantity, unit_price)
products(sku_id, product_name, category, is_active)
refunds(refund_id, order_id, refund_amount, refund_reason, refund_date)""",
        metadata_json='{"currency":"CNY","date_timezone":"Asia/Shanghai","status_values":["paid","shipped","refunded","cancelled"]}',
        sample_rows="""orders: O1001,U01,2026-03-01,paid,299.00
order_items: O1001,SKU88,1,299.00
products: SKU88,智能手环,数码,true
refunds: R9001,O1002,99.00,质量问题,2026-03-03""",
        few_shot_examples="""Q: O1001 的订单状态是什么？
A: O1001 在 orders.status 中为 paid。""",
    )
    store.save_project_context(context)
    samples = [
        EvalSample(
            question="O1001 的订单状态是什么？",
            question_type="事实核对",
            difficulty="低",
            expected_scope="应查询 orders.status。",
            reference_answer="O1001 的订单状态为 paid。",
            expected_evidence="orders.status",
            tags=["demo", "订单"],
            source_context_refs=["demo_rows.csv", "demo_schema.sql"],
            generation_method="seed",
            review_status="已通过",
        ),
        EvalSample(
            question="如果用户询问退款金额，系统需要引用哪张表？",
            question_type="业务规则",
            difficulty="中",
            expected_scope="应基于业务规则说明 refunds 表。",
            reference_answer="查询退款金额时必须引用 refunds 表。",
            expected_evidence="查询退款金额时必须引用 refunds 表",
            tags=["demo", "退款"],
            source_context_refs=["business_rules"],
            generation_method="seed",
            review_status="已通过",
        ),
        EvalSample(
            question="当问题没有时间范围时，系统应该直接给出某个月的统计吗？",
            question_type="边界条件",
            difficulty="中",
            expected_scope="应说明时间范围不明确，不能擅自假设某个月。",
            reference_answer="不应直接假设某个月；应说明默认范围或要求用户补充时间范围。",
            expected_evidence="用户问题没有时间范围时，应说明默认使用全部可见数据或要求补充时间范围",
            tags=["demo", "边界"],
            source_context_refs=["business_rules"],
            generation_method="seed",
            review_status="已通过",
        ),
    ]
    store.save_test_cases(context.project_id, samples)
