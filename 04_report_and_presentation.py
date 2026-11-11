from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
def decide(ci_low, ci_high, mde, guardrails_ok=True):
    if ci_high < 0:
        return "لا نطلق، لأن النسخة الجديدة أضرت بالمقياس الرئيسي."
    if ci_low > 0 and not guardrails_ok:
        return "لا نطلق الآن، لأن المقياس الرئيسي تحسن لكن أحد مقاييس الحماية ساء."
    if ci_low > 0:
        return "نطلق، لأن الأثر إيجابي بثقة ومقاييس الحماية سليمة."
    if ci_high < mde:
        return "لا نطلق، لأننا استبعدنا وجود أثر بالحجم الذي يهمنا."
    return "النتيجة غير حاسمة؛ نعيد تصميم التجربة أو نجمع أدلة إضافية."

def main():
    out = ROOT / "outputs" / "results"
    cfg = json.loads((out / "experiment_config.json").read_text(encoding="utf-8"))["config"]
    results = json.loads((out / "analysis_results.json").read_text(encoding="utf-8"))
    p, cuped, guard = results["primary"], results["cuped"], results["guardrail"]
    guard_ok = guard["status"] != "significant_harm"
    decision = decide(p["أدنى 95%"], p["أعلى 95%"], cfg["mde_abs"], guard_ok)
    annual_users = 50_000
    annual_impact = annual_users * p["الفرق"]
    report = f"""# تقرير الصفحة الواحدة | أثر توصيات الذكاء الاصطناعي على دقة قرارات الأعمال

**برنامج التجارب واختبارات A/B والاستدلال السببي — أكاديمية سدايا**

## السؤال والتصميم
هل تزيد توصيات AI دقة قرارات الأعمال؟ A: لوحة بيانات فقط؛ B: لوحة بيانات مع توصيات AI. التوزيع عشوائي على مستوى المستخدم بنسبة 50/50، والبيانات **محاكاة وليست مستخدمين حقيقيين**.

## حجم العينة
21 يومًا، {results['n']:,} مستخدم؛ مستوى الدلالة 5%، القوة المستهدفة 80%، وMDE = 5 نقاط مئوية.

## النتائج
- دقة القرار: A = {results['accuracy']['A']:.2%}، B = {results['accuracy']['B']:.2%}.
- الفرق: {p['الفرق']*100:+.2f} نقطة مئوية، فترة الثقة 95% [{p['أدنى 95%']*100:.2f}, {p['أعلى 95%']*100:.2f}]، p = {p['p']:.4f}.
- CUPED: {cuped['الفرق']*100:+.2f} نقطة، فترة الثقة 95% [{cuped['أدنى 95%']*100:.2f}, {cuped['أعلى 95%']*100:.2f}].
- الحماية (أخطاء عالية الأثر): A = {guard['A']:.2%}، B = {guard['B']:.2%}؛ {guard['status']}.
- إسقاط سنوي افتراضي لـ {annual_users:,} قرار: {annual_impact:,.0f} قرار صحيح إضافي، غير مؤكد إحصائيًا.

## القرار والتوصية
**{decision}** نوصي بتحسين التوصيات وإعادة اختبارها. لا يوجد دليل كافٍ على تحسن دال في المقياس الرئيسي.

## القيود
المحاكاة لا تثبت أثرًا في الواقع؛ ومقياس الأخطاء عالية الأثر مشتق من صحة القرار، لذا ليس مستقلًا عنها. عدم اكتشاف ضرر لا يثبت انعدام الضرر.
"""
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "One_Page_Report.md").write_text(report, encoding="utf-8")
    slides = f"""# عرض المشروع — 10 دقائق (مخطط محتوى)

1. الغلاف — عنوان المشروع واسم البرنامج (0:30)
2. الملخص التنفيذي والقرار (1:00)
3. السؤال والفرضية وتصميم A/B (1:30)
4. حجم العينة والمدة (1:00)
5. النتائج الرئيسية: A {results['accuracy']['A']:.2%} وB {results['accuracy']['B']:.2%} (2:00)
6. CUPED والمقاييس الثانوية (1:30)
7. مقاييس الحماية والقيود (1:30)
8. القرار النهائي والتوصية: {decision} (1:00)

النتائج محاكاة افتراضية. رسوم العرض في outputs/charts.
"""
    (ROOT / "reports" / "Presentation_Outline.md").write_text(slides, encoding="utf-8")
    (out / "final_decision.json").write_text(json.dumps({"decision": decision, "annual_impact_estimate": annual_impact, "note": "simulation only"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("04 | Report and presentation outline generated:", decision)

if __name__ == "__main__":
    main()
