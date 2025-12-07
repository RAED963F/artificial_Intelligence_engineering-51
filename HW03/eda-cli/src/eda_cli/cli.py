from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))

@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    
    # ⭐⭐⭐ НОВЫЕ ПАРАМЕТРЫ:
    # 1. Порог для высокой доли нулей
    zero_threshold: float = typer.Option(
        30.0, 
        help="Порог процента нулей для пометки столбца (по умолчанию 30%)."
    ),
    
    # 2. Включение расширенного анализа качества
    advanced_quality_check: bool = typer.Option(
        False,
        help="Включить расширенный анализ качества данных (постоянные столбцы, выбросы и т.д.)."
    ),
    
    # 3. Порог для выбросов (IQR множитель)
    iqr_multiplier: float = typer.Option(
        1.5,
        help="Множитель IQR для определения выбросов (по умолчанию 1.5)."
    ),
    
    # 4. Минимальная доля пропусков для детального отчёта
    min_missing_for_detail: float = typer.Option(
        5.0,
        help="Минимальный процент пропусков для детального отчёта по столбцу."
    ),
    
    # 5. Максимальное количество топ-категорий для отображения
    top_k_categories: int = typer.Option(
        10,
        help="Количество топ-значений для категориальных признаков."
    ),
    
    # 6. Включение/отключение матрицы корреляций
    include_correlation: bool = typer.Option(
        True,
        help="Включать ли матрицу корреляций в отчёт."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт с расширенными возможностями.
    Включает новые параметры для кастомизации анализа.
    """
    # Таймер для измерения времени выполнения
    import time
    start_time = time.time()
    
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Создаём подкаталоги
    figures_dir = out_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    typer.echo("=" * 60)
    typer.echo("📊 ЗАПУСК РАСШИРЕННОГО EDA-АНАЛИЗА")
    typer.echo("=" * 60)
    typer.echo(f"📁 Файл: {path}")
    typer.echo(f"📂 Выходной каталог: {out_dir}")
    
    if advanced_quality_check:
        typer.echo("🔍 Расширенный анализ качества: ВКЛЮЧЕН")
        typer.echo(f"⚙️  Параметры: zero_threshold={zero_threshold}%, iqr_multiplier={iqr_multiplier}")
    
    # 1. Загрузка данных
    typer.echo("\n⏳ Загрузка данных...")
    try:
        df = _load_csv(Path(path), sep=sep, encoding=encoding)
        typer.echo(f"✅ Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    except Exception as e:
        typer.echo(f"❌ Ошибка загрузки файла: {e}", err=True)
        raise typer.Exit(1)
    
    # 2. Базовый анализ
    typer.echo("⏳ Вычисление базовой статистики...")
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    
    # 3. Корреляционная матрица (если включена)
    corr_df = pd.DataFrame()
    if include_correlation:
        typer.echo("⏳ Вычисление корреляционной матрицы...")
        corr_df = correlation_matrix(df)
    
    # 4. Топ-категории с новым параметром top_k_categories
    typer.echo(f"⏳ Анализ категориальных признаков (top-{top_k_categories})...")
    top_cats = top_categories(df, top_k=top_k_categories)
    
    # 5. Расширенный анализ качества (если включен)
    if advanced_quality_check:
        typer.echo("⏳ Расширенный анализ качества данных...")
        # Расширенная версия compute_quality_flags
        quality_flags = compute_quality_flags_extended(
            summary=summary,
            missing_df=missing_df,
            df=df,
            zero_threshold=zero_threshold,
            iqr_multiplier=iqr_multiplier,
            verbose=True
        )
    else:
        quality_flags = compute_quality_flags(summary, missing_df)
    
    # 6. Сохранение табличных данных
    typer.echo("⏳ Сохранение табличных данных...")
    
    # 6.1. Основная статистика
    summary_df.to_csv(out_root / "summary.csv", index=False)
    
    # 6.2. Пропущенные значения
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    
    # 6.3. Корреляционная матрица
    if include_correlation and not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    
    # 6.4. Топ-категории
    if top_cats:
        save_top_categories_tables(top_cats, out_root / "top_categories")
    
    # 6.5. Расширенные флаги качества (если есть)
    if advanced_quality_check:
        quality_flags_path = out_root / "quality_flags.json"
        import json
        with open(quality_flags_path, 'w', encoding='utf-8') as f:
            # Преобразуем numpy типы для сериализации
            def convert_for_json(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return obj
            
            serializable_flags = {}
            for key, value in quality_flags.items():
                if isinstance(value, (list, dict)):
                    serializable_flags[key] = json.loads(
                        json.dumps(value, default=convert_for_json)
                    )
                else:
                    serializable_flags[key] = convert_for_json(value)
            
            json.dump(serializable_flags, f, indent=2, ensure_ascii=False)
    
    # 7. Генерация Markdown-отчёта
    typer.echo("⏳ Генерация Markdown-отчёта...")
    md_path = out_root / "report.md"
    
    with md_path.open("w", encoding="utf-8") as f:
        # ==================== ЗАГОЛОВОК ====================
        f.write(f"# 📊 EDA-отчёт: Анализ данных\n\n")
        f.write(f"**Исходный файл:** `{Path(path).name}`\n")
        f.write(f"**Дата анализа:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Время выполнения:** {time.time() - start_time:.2f} секунд\n\n")
        
        # ==================== ПАРАМЕТРЫ АНАЛИЗА ====================
        f.write("## ⚙️ Параметры анализа\n\n")
        f.write("| Параметр | Значение |\n")
        f.write("|----------|----------|\n")
        f.write(f"| Разделитель CSV | `{sep}` |\n")
        f.write(f"| Кодировка | `{encoding}` |\n")
        f.write(f"| Макс. гистограмм | `{max_hist_columns}` |\n")
        f.write(f"| Топ-K категорий | `{top_k_categories}` |\n")
        f.write(f"| Вкл. корреляцию | `{include_correlation}` |\n")
        
        if advanced_quality_check:
            f.write(f"| Расширенный анализ | **Включён** |\n")
            f.write(f"| Порог нулей | `{zero_threshold}%` |\n")
            f.write(f"| Множитель IQR | `{iqr_multiplier}` |\n")
            f.write(f"| Порог детальных пропусков | `{min_missing_for_detail}%` |\n")
        else:
            f.write(f"| Расширенный анализ | Выключен |\n")
        
        f.write("\n")
        
        # ==================== ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ====================
        f.write("## 📈 Основные характеристики\n\n")
        f.write(f"- **Строк (записей):** `{summary.n_rows}`\n")
        f.write(f"- **Столбцов (признаков):** `{summary.n_cols}`\n")
        f.write(f"- **Числовых столбцов:** `{summary.n_numeric}`\n")
        f.write(f"- **Категориальных столбцов:** `{summary.n_categorical}`\n")
        f.write(f"- **Столбцов с датой/временем:** `{summary.n_datetime}`\n")
        f.write(f"- **Других типов:** `{summary.n_other}`\n\n")
        
        # ==================== КАЧЕСТВО ДАННЫХ ====================
        f.write("## 🔍 Качество данных\n\n")
        
        # Общий score
        f.write(f"### Общая оценка качества\n")
        f.write(f"**Score:** `{quality_flags.get('quality_score', 0):.2f}` из 1.0\n\n")
        
        # Базовые флаги
        f.write(f"### Базовые проверки\n")
        f.write(f"- Слишком мало строк (<100): **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много столбцов (>100): **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Макс. доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком много пропусков (>50%): **{quality_flags['too_many_missing']}**\n\n")
        
        # Расширенные флаги (если включены)
        if advanced_quality_check:
            f.write(f"### Расширенные проверки\n")
            
            # Постоянные столбцы
            if quality_flags.get('has_constant_columns', False):
                f.write(f"- ⚠️ **Постоянные столбцы:** ДА ({quality_flags.get('n_constant_columns', 0)} шт.)\n")
                if quality_flags.get('constant_columns'):
                    f.write("  Столбцы:\n")
                    for col_info in quality_flags['constant_columns'][:5]:  # показываем первые 5
                        f.write(f"  - `{col_info['column']}` = `{col_info['value']}`\n")
                    if len(quality_flags['constant_columns']) > 5:
                        f.write(f"  ... и ещё {len(quality_flags['constant_columns']) - 5} столбцов\n")
            else:
                f.write(f"- ✅ **Постоянные столбцы:** Нет\n")
            
            # Выбросы
            if quality_flags.get('has_outliers', False):
                f.write(f"- ⚠️ **Выбросы в данных:** ДА ({quality_flags.get('n_outlier_columns', 0)} столбцов)\n")
                if quality_flags.get('outlier_columns'):
                    f.write("  Проблемные столбцы:\n")
                    for col_info in quality_flags['outlier_columns'][:3]:  # показываем первые 3
                        f.write(f"  - `{col_info['column']}`: {col_info['n_outliers']} выбросов ({col_info['outlier_percentage']}%)\n")
            else:
                f.write(f"- ✅ **Выбросы в данных:** Нет\n")
            
            # Дубликаты ID
            if quality_flags.get('has_id_duplicates', False):
                f.write(f"- ⚠️ **Дубликаты ID:** ДА\n")
                if quality_flags.get('id_duplicate_issues'):
                    for issue in quality_flags['id_duplicate_issues'][:2]:
                        f.write(f"  - `{issue['column']}`: {issue['n_duplicates']} дубликатов ({issue['duplicate_percentage']}%)\n")
            else:
                f.write(f"- ✅ **Дубликаты ID:** Нет\n")
            
            # Высокая доля нулей
            if quality_flags.get('has_high_zero_columns', False):
                f.write(f"- ⚠️ **Высокая доля нулей (> {zero_threshold}%):** ДА\n")
                if quality_flags.get('high_zero_columns'):
                    for col_info in quality_flags['high_zero_columns'][:3]:
                        f.write(f"  - `{col_info['column']}`: {col_info['zero_count']} нулей ({col_info['zero_percentage']}%)\n")
            else:
                f.write(f"- ✅ **Высокая доля нулей:** Нет\n")
            
            f.write("\n")
        
        # ==================== ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ====================
        f.write("## 🕳️ Пропущенные значения\n\n")
        
        if missing_df.empty:
            f.write("✅ **Пропущенных значений нет.**\n\n")
        else:
            # Общая статистика
            total_missing = missing_df['missing_count'].sum()
            total_cells = summary.n_rows * summary.n_cols
            overall_missing_percent = (total_missing / total_cells) * 100
            
            f.write(f"### Общая статистика\n")
            f.write(f"- Всего пропусков: `{total_missing}`\n")
            f.write(f"- Доля пропусков в датасете: `{overall_missing_percent:.2f}%`\n")
            f.write(f"- Столбцов с пропусками: `{len(missing_df[missing_df['missing_count'] > 0])}`\n\n")
            
            # Проблемные столбцы (с учётом min_missing_for_detail)
            problem_cols = missing_df[missing_df['missing_share'] * 100 >= min_missing_for_detail]
            
            if not problem_cols.empty:
                f.write(f"### ⚠️ Столбцы с пропусками > {min_missing_for_detail}%\n\n")
                f.write("| Столбец | Пропусков | Доля пропусков |\n")
                f.write("|---------|-----------|----------------|\n")
                
                for idx, row in problem_cols.iterrows():
                    f.write(f"| `{idx}` | {row['missing_count']} | {row['missing_share']:.2%} |\n")
                
                f.write("\n")
            
            f.write("> Полная таблица пропусков в файле `missing.csv`\n\n")
        
        # ==================== КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ====================
        f.write("## 📊 Категориальные признаки\n\n")
        
        if not top_cats:
            f.write("Категориальные признаки не найдены.\n\n")
        else:
            f.write(f"Найдено `{len(top_cats)}` категориальных признаков.\n")
            f.write(f"Для каждого показаны топ-`{top_k_categories}` значений.\n\n")
            
            # Пример для первых 3 столбцов
            for i, (col_name, categories) in enumerate(top_cats.items()):
                if i >= 3:  # показываем только первые 3
                    remaining = len(top_cats) - 3
                    f.write(f"\n... и ещё `{remaining}` категориальных признаков.\n")
                    break
                
                f.write(f"### `{col_name}`\n")
                f.write(f"Уникальных значений: `{categories['n_unique']}`\n\n")
                
                if categories['top_values']:
                    f.write("| Значение | Количество | Доля |\n")
                    f.write("|----------|------------|------|\n")
                    
                    for value, count in categories['top_values'].items():
                        percentage = (count / summary.n_rows) * 100
                        f.write(f"| `{value}` | {count} | {percentage:.1f}% |\n")
                
                f.write("\n")
            
            f.write("> Подробные таблицы в папке `top_categories/`\n\n")
        
        # ==================== КОРРЕЛЯЦИИ ====================
        if include_correlation:
            f.write("## 🔗 Корреляционная матрица\n\n")
            
            if corr_df.empty or len(corr_df) <= 1:
                f.write("Недостаточно числовых столбцов для корреляционного анализа.\n\n")
            else:
                f.write(f"Размер матрицы: `{corr_df.shape[0]}×{corr_df.shape[1]}`\n\n")
                
                # Самые сильные корреляции
                strong_correlations = []
                for i in range(len(corr_df.columns)):
                    for j in range(i+1, len(corr_df.columns)):
                        corr_value = corr_df.iloc[i, j]
                        if abs(corr_value) > 0.7:  # сильная корреляция
                            strong_correlations.append((
                                corr_df.columns[i],
                                corr_df.columns[j],
                                corr_value
                            ))
                
                if strong_correlations:
                    f.write("### Сильные корреляции (|r| > 0.7)\n\n")
                    f.write("| Признак 1 | Признак 2 | Коэффициент |\n")
                    f.write("|-----------|-----------|-------------|\n")
                    
                    for col1, col2, corr_val in strong_correlations[:10]:  # первые 10
                        f.write(f"| `{col1}` | `{col2}` | {corr_val:.3f} |\n")
                    
                    if len(strong_correlations) > 10:
                        f.write(f"| ... | ... | ... |\n")
                    
                    f.write("\n")
                
                f.write("> Полная матрица в файле `correlation.csv`\n\n")
        
        # ==================== ВИЗУАЛИЗАЦИИ ====================
        f.write("## 🎨 Визуализации\n\n")
        
        f.write("### Гистограммы числовых признаков\n")
        f.write(f"Сгенерировано гистограмм: до `{max_hist_columns}` шт.\n")
        f.write("Файлы: `hist_*.png`\n\n")
        
        f.write("### Матрица пропусков\n")
        f.write("Файл: `missing_matrix.png`\n\n")
        
        if include_correlation and not corr_df.empty:
            f.write("### Тепловая карта корреляций\n")
            f.write("Файл: `correlation_heatmap.png`\n\n")
        
        # ==================== РЕКОМЕНДАЦИИ ====================
        f.write("## 💡 Рекомендации\n\n")
        
        recommendations = []
        
        if quality_flags.get('too_many_missing', False):
            recommendations.append("**Пропуски:** Рассмотрите импутацию или удаление столбцов с высокой долей пропусков.")
        
        if quality_flags.get('too_few_rows', False):
            recommendations.append("**Мало данных:** Для надёжного анализа рекомендуется собрать больше данных.")
        
        if advanced_quality_check:
            if quality_flags.get('has_constant_columns', False):
                recommendations.append("**Постоянные столбцы:** Удалите столбцы со всеми одинаковыми значениями.")
            
            if quality_flags.get('has_outliers', False):
                recommendations.append("**Выбросы:** Проверьте выбросы на предмет ошибок в данных.")
            
            if quality_flags.get('has_id_duplicates', False):
                recommendations.append("**Дубликаты ID:** Исследуйте дубликаты идентификаторов.")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        else:
            f.write("✅ Качество данных в целом хорошее. Продолжайте анализ!\n")
        
        f.write("\n")
        
        # ==================== ФАЙЛЫ ОТЧЁТА ====================
        f.write("## 📁 Файлы отчёта\n\n")
        f.write("| Файл | Описание |\n")
        f.write("|------|----------|\n")
        f.write(f"| `{md_path.name}` | Этот отчёт (Markdown) |\n")
        f.write(f"| `summary.csv` | Детальная статистика по столбцам |\n")
        
        if not missing_df.empty:
            f.write(f"| `missing.csv` | Таблица пропусков по столбцам |\n")
            f.write(f"| `missing_matrix.png` | Визуализация пропусков |\n")
        
        if include_correlation and not corr_df.empty:
            f.write(f"| `correlation.csv` | Матрица корреляций |\n")
            f.write(f"| `correlation_heatmap.png` | Тепловая карта корреляций |\n")
        
        if top_cats:
            f.write(f"| `top_categories/*.csv` | Топ-значения категориальных признаков |\n")
        
        if advanced_quality_check:
            f.write(f"| `quality_flags.json` | Расширенные флаги качества |\n")
        
        f.write(f"| `hist_*.png` | Гистограммы числовых признаков |\n")
        
        f.write("\n")
        f.write("---\n")
        f.write("*Отчёт сгенерирован автоматически с помощью `eda-cli`*\n")
        f.write(f"*Версия: 1.1.0 | Расширенный анализ: {'Да' if advanced_quality_check else 'Нет'}*\n")
    
    # 8. Генерация визуализаций
    typer.echo("⏳ Генерация визуализаций...")
    
    # 8.1. Гистограммы
    plot_histograms_per_column(df, figures_dir, max_columns=max_hist_columns)
    
    # 8.2. Матрица пропусков
    plot_missing_matrix(df, figures_dir / "missing_matrix.png")
    
    # 8.3. Тепловая карта корреляций (если включена)
    if include_correlation and not corr_df.empty:
        plot_correlation_heatmap(df, figures_dir / "correlation_heatmap.png")
    
    # 9. Финальное сообщение
    execution_time = time.time() - start_time
    
    typer.echo("\n" + "=" * 60)
    typer.echo("✅ АНАЛИЗ ЗАВЕРШЁН УСПЕШНО!")
    typer.echo("=" * 60)
    typer.echo(f"📊 Основные результаты:")
    typer.echo(f"   • Отчёт: {md_path}")
    typer.echo(f"   • Столбцов проанализировано: {summary.n_cols}")
    typer.echo(f"   • Оценка качества: {quality_flags.get('quality_score', 0):.2f}/1.0")
    
    if advanced_quality_check:
        typer.echo(f"   • Расширенные флаги: quality_flags.json")
    
    typer.echo(f"\n⏱️  Время выполнения: {execution_time:.2f} секунд")
    typer.echo(f"📁 Все файлы сохранены в: {out_root}")
    typer.echo("=" * 60)

if __name__ == "__main__":
    app()
