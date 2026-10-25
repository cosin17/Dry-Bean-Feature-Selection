"""干豆分类：训练、评估、结果绘图与模型保存。运行方法见README。"""

from pathlib import Path
from datetime import datetime
from functools import partial
from itertools import combinations
import json
import time
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 图片保存到文件，运行时不弹出窗口。
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay, log_loss
from threadpoolctl import threadpool_limits

from data_loader import load_dry_bean_data, DATA_SHA256

# 在这里查看或修改实验参数；不要根据测试分数反复修改参数。
RANDOM_SEED = 20261025
FEATURE_COUNTS = [4, 8, 12, 16]
NUMBER_OF_FOLDS = 5
F1_TOLERANCE = 0.01
PROJECT_DIRECTORY = Path(__file__).resolve().parent
data_directory = PROJECT_DIRECTORY / "data"
results_directory = PROJECT_DIRECTORY / "results"
figures_directory = results_directory / "figures"
models_directory = PROJECT_DIRECTORY / "models"


def save_figure(figure, filename):
    """保存当前图并释放内存；可在VS Code中打开PNG查看。"""
    figure.savefig(figures_directory / filename, dpi=600)
    plt.close(figure)


def save_results(table, filename, run_id):
    """用run_id区分每次实验，CSV追加保存，不创建重复结果目录。"""
    table = table.copy()
    table.insert(0, "run_id", run_id)
    path = results_directory / filename
    if path.exists():
        # 兼容旧版结果：新增指标列时，旧实验对应位置留空。
        old_columns = pd.read_csv(path, nrows=0).columns.tolist()
        if old_columns != table.columns.tolist():
            table = pd.concat([pd.read_csv(path), table], ignore_index=True)
            table.to_csv(path, index=False)
            return
    table.to_csv(path, mode="a", header=not path.exists(), index=False)


def log_progress(message, run_id, show=True):
    if show:
        print(message, flush=True)
    with (results_directory / "training_log.txt").open("a", encoding="utf-8") as file:
        file.write(f"[{run_id}] {message}\n")


def evaluate_fold(model, training_features, training_labels, validation_features,
                  validation_labels, description, run_id):
    """记录一次真实训练/验证结果。Fold是独立划分，不是Epoch。"""
    training_predictions = model.predict(training_features)
    predicted = model.predict(validation_features)
    classes = model.classes_
    scores = {
        "accuracy": accuracy_score(validation_labels, predicted),
        "macro_f1": f1_score(validation_labels, predicted, average="macro"),
        "train_accuracy": accuracy_score(training_labels, training_predictions),
        "train_loss": log_loss(training_labels, model.predict_proba(training_features), labels=classes),
        "validation_loss": log_loss(validation_labels, model.predict_proba(validation_features), labels=classes),
    }
    log_progress(
        f"{description} | Train_acc: {scores['train_accuracy']:.1%}, "
        f"Train_loss: {scores['train_loss']:.3f}, Val_acc: {scores['accuracy']:.1%}, "
        f"Val_loss: {scores['validation_loss']:.3f}, Val_F1: {scores['macro_f1']:.4f}", run_id)
    return predicted, scores


def create_model(model_name, number_of_features=None):
    classifiers = {
        "LR": LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=2),
        "Dummy": DummyClassifier(strategy="most_frequent")}
    steps = [("scale", StandardScaler())]
    if number_of_features is not None:
        score_function = partial(mutual_info_classif, random_state=RANDOM_SEED)
        steps.append(("select", SelectKBest(score_func=score_function, k=number_of_features)))
    steps.append(("classifier", classifiers[model_name]))
    return Pipeline(steps)


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    started_at = time.perf_counter()
    for directory in [data_directory, results_directory, figures_directory, models_directory]:
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False})
    log_progress(f"开始实验 {run_id}，seed={RANDOM_SEED}，5折；LR的C=1，KNN邻居数=5，RF树数=100", run_id)
    log_progress("训练进度中Val表示验证集；loss是预测概率的交叉熵，越小越好。测试集最后才评价。", run_id)

    # 1. 读取、去重和划分数据
    raw_data = load_dry_bean_data(data_directory)
    feature_names = raw_data.columns.drop("Class").tolist()
    print("原始数据形状：", raw_data.shape)
    print("缺失值数量：", raw_data.isna().sum().sum())
    # 相同输入如果对应不同标签，就不能悄悄保留其中一个标签。
    assert raw_data.groupby(feature_names)["Class"].nunique().max() == 1
    clean_data = raw_data.drop_duplicates(subset=feature_names).copy()
    clean_data.index.name = "sample_id"
    print("删除重复行：", len(raw_data) - len(clean_data))

    features = clean_data[feature_names]
    labels = clean_data["Class"]
    features_development, features_test, labels_development, labels_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)
    class_names = sorted(labels_development.unique())
    print("开发集：", len(features_development), "；测试集：", len(features_test))

    splitter = StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(splitter.split(features_development, labels_development))
    clean_data["split"] = "test"
    clean_data.loc[features_development.index, "split"] = "development"
    clean_data["cv_fold"] = -1
    for fold_number, (_, validation_indices) in enumerate(folds, start=1):
        clean_data.loc[features_development.iloc[validation_indices].index, "cv_fold"] = fold_number
    clean_data.to_csv(data_directory / "dry_bean_clean.csv")

    # 2. 查看开发集数据
    class_counts = labels_development.value_counts().reindex(class_names)
    figure, axis = plt.subplots(figsize=(7.1, 4.0), layout="constrained")
    axis.bar(class_counts.index, class_counts.values, color="#3568A6")
    axis.set(xlabel="Bean class", ylabel="Development sample count")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    save_figure(figure, "class_distribution.png")

    correlations = features_development.corr()
    figure, axis = plt.subplots(figsize=(8.4, 7.0), layout="constrained")
    image = axis.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label="Pearson correlation")
    axis.set(xticks=range(16), yticks=range(16), xticklabels=feature_names, yticklabels=feature_names)
    plt.setp(axis.get_xticklabels(), rotation=90, ha="right", va="center", rotation_mode="anchor", fontsize=7.5)
    plt.setp(axis.get_yticklabels(), fontsize=7.5)
    save_figure(figure, "feature_correlation.png")

    # 3. LR和KNN的特征数量对比
    validation_records = []
    validation_predictions = {}
    selected_feature_sets = {count: [] for count in FEATURE_COUNTS if count < 16}

    with threadpool_limits(limits=2):
        for model_name in ["LR", "KNN"]:
            for count in FEATURE_COUNTS:
                fold_scores = []
                for fold_number, (training_indices, validation_indices) in enumerate(folds, start=1):
                    model = create_model(model_name, None if count == 16 else count)
                    model.fit(features_development.iloc[training_indices], labels_development.iloc[training_indices])
                    actual = labels_development.iloc[validation_indices]
                    predicted, scores = evaluate_fold(
                        model, features_development.iloc[training_indices], labels_development.iloc[training_indices],
                        features_development.iloc[validation_indices], actual,
                        f"Model: {model_name:3s}, Features: {count:2d}, Fold: {fold_number}/{NUMBER_OF_FOLDS}", run_id)
                    score = scores["macro_f1"]
                    fold_scores.append(score)
                    chosen_features = (np.array(feature_names)[model.named_steps["select"].get_support()].tolist()
                                       if count < 16 else feature_names)
                    if model_name == "LR" and count < 16:
                        selected_feature_sets[count].append(chosen_features)
                    validation_records.append({"model": model_name, "selection": "MI" if count < 16 else "All",
                        "feature_count": count, "fold": fold_number, "seed": RANDOM_SEED, "subset_seed": -1,
                        "features": "|".join(chosen_features), **scores})
                    key = f"{model_name}_{count}_fold_{fold_number}"
                    validation_predictions[key + "_ids"] = actual.index.to_numpy()
                    validation_predictions[key + "_true"] = actual.to_numpy(dtype=str)
                    validation_predictions[key + "_predicted"] = predicted.astype(str)
                log_progress(f"{model_name}，{count:2d}个特征，5折宏F1：{np.mean(fold_scores):.4f}", run_id)

    validation_table = pd.DataFrame(validation_records)
    save_results(validation_table, "cross_validation_results.csv", run_id)
    np.savez_compressed(results_directory / "validation_predictions.npz", **validation_predictions)
    model_comparison = validation_table.groupby(["model", "feature_count"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), fold_std=("macro_f1", "std"), accuracy=("accuracy", "mean"))
    save_results(model_comparison, "model_comparison.csv", run_id)
    print(model_comparison.round(4))

    figure, axis = plt.subplots(figsize=(7.1, 4.1), layout="constrained")
    for model_name, color, marker in [("LR", "#3568A6", "o"), ("KNN", "#C47C38", "s")]:
        rows = model_comparison[model_comparison["model"] == model_name]
        axis.errorbar(rows["feature_count"], rows["macro_f1"], yerr=rows["fold_std"],
                      label=model_name, color=color, marker=marker, capsize=3)
    axis.set(xlabel="Number of retained features", ylabel="Validation macro F1", xticks=FEATURE_COUNTS)
    axis.legend()
    axis.margins(y=0.2)
    save_figure(figure, "feature_count_comparison.png")

    print("误差条是5折分数的标准差，只描述划分波动，不是独立实验置信区间。")

    # 4. 只用验证分数确定特征数量
    selected_counts = {}
    for model_name in ["LR", "KNN"]:
        rows = model_comparison[model_comparison["model"] == model_name]
        full_score = rows.loc[rows["feature_count"] == 16, "macro_f1"].iloc[0]
        acceptable = rows[rows["macro_f1"] >= full_score - F1_TOLERANCE]
        selected_counts[model_name] = int(acceptable["feature_count"].min())
    print("开发集选出的特征数：", selected_counts)

    experiment_settings = {"run_id": run_id, "status": "selection_fixed", "seed": RANDOM_SEED,
        "folds": NUMBER_OF_FOLDS, "feature_counts": FEATURE_COUNTS, "f1_tolerance": F1_TOLERANCE,
        "fixed_parameters": {"LR_C": 1.0, "KNN_neighbors": 5, "RF_trees": 100},
        "random_subset_seeds": [1, 2, 3], "selected_counts": selected_counts,
        "test_used_for_selection": False, "data_sha256": DATA_SHA256,
        "raw_rows": len(raw_data), "clean_rows": len(clean_data),
        "development_rows": len(features_development), "test_rows": len(features_test)}
    settings_path = results_directory / "experiment_settings.json"
    settings_path.write_text(json.dumps(experiment_settings, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. 随机特征对照
    random_records = []
    with threadpool_limits(limits=2):
        for count in [4, 8, 12]:
            for subset_seed in [1, 2, 3]:
                random_features = np.random.default_rng(subset_seed).choice(feature_names, count, replace=False).tolist()
                for fold_number, (training_indices, validation_indices) in enumerate(folds, start=1):
                    model = create_model("KNN")  # 列已随机选定，这里不再做MI筛选
                    model.fit(features_development.iloc[training_indices][random_features], labels_development.iloc[training_indices])
                    actual = labels_development.iloc[validation_indices]
                    predicted, scores = evaluate_fold(
                        model, features_development.iloc[training_indices][random_features], labels_development.iloc[training_indices],
                        features_development.iloc[validation_indices][random_features], actual,
                        f"Model: KNN, Random: {subset_seed}, Features: {count:2d}, Fold: {fold_number}/{NUMBER_OF_FOLDS}", run_id)
                    random_records.append({"model": "KNN", "selection": "Random", "feature_count": count,
                        "fold": fold_number, "seed": RANDOM_SEED, "subset_seed": subset_seed,
                        "features": "|".join(random_features), **scores})
                    key = f"random_{count}_seed_{subset_seed}_fold_{fold_number}"
                    validation_predictions[key + "_ids"] = actual.index.to_numpy()
                    validation_predictions[key + "_true"] = actual.to_numpy(dtype=str)
                    validation_predictions[key + "_predicted"] = predicted.astype(str)
                log_progress(f"随机对照：{count}个特征，子集种子{subset_seed}，5折完成", run_id)
    random_table = pd.DataFrame(random_records)
    save_results(random_table, "cross_validation_results.csv", run_id)
    np.savez_compressed(results_directory / "validation_predictions.npz", **validation_predictions)
    mi_scores = validation_table[(validation_table["model"] == "KNN") & (validation_table["feature_count"] < 16)]
    random_scores = random_table.groupby(["feature_count", "fold"], as_index=False)["macro_f1"].mean()
    paired_scores = mi_scores.merge(random_scores, on=["feature_count", "fold"], suffixes=("_mi", "_random"))
    paired_scores["difference"] = paired_scores["macro_f1_mi"] - paired_scores["macro_f1_random"]
    print(paired_scores.groupby("feature_count")["difference"].agg(["mean", "std"]).round(4))

    difference_summary = paired_scores.groupby("feature_count")["difference"].agg(["mean", "std"])
    figure, axis = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    axis.axhline(0, color="#78818A", linestyle="--", linewidth=0.8)
    axis.errorbar(difference_summary.index, difference_summary["mean"], yerr=difference_summary["std"],
                  fmt="o-", color="#3568A6", capsize=4)
    axis.set(xlabel="Number of retained features", ylabel="Macro F1 difference (MI - random)", xticks=[4, 8, 12])
    axis.margins(x=0.2, y=0.3)
    save_figure(figure, "random_feature_comparison.png")

    # 6. 特征选择稳定性
    feature_frequency = pd.DataFrame({"feature": feature_names})
    stability_scores = {}
    for count in [4, 8, 12]:
        feature_frequency[f"top_{count}"] = [sum(name in names for names in selected_feature_sets[count])
                                               for name in feature_names]
        pairs = combinations([set(names) for names in selected_feature_sets[count]], 2)
        stability_scores[count] = [len(first & second) / len(first | second) for first, second in pairs]
    save_results(feature_frequency, "feature_selection_frequency.csv", run_id)
    print("各维数平均Jaccard相似度：", {count: round(float(np.mean(scores)), 3) for count, scores in stability_scores.items()})
    figure, axis = plt.subplots(figsize=(6.4, 5.7), layout="constrained")
    image = axis.imshow(feature_frequency.iloc[:, 1:].to_numpy(), cmap="Blues", vmin=0, vmax=5, aspect="auto")
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03, label="Selected folds")
    axis.set(xticks=[0, 1, 2], xticklabels=[4, 8, 12], yticks=range(16), yticklabels=feature_names,
             xlabel="Number of retained features")
    save_figure(figure, "feature_selection_frequency.png")

    # 7. 最终测试
    log_progress("方案已经确定，开始最终测试评价。测试结果不用于改变参数或特征数。", run_id)
    test_models = {"LR_full": ("LR", 16), "LR_selected": ("LR", selected_counts["LR"]),
                   "KNN_full": ("KNN", 16), "KNN_selected": ("KNN", selected_counts["KNN"]),
                   "RF": ("RF", 16), "Dummy": ("Dummy", 16)}
    test_records, prediction_tables, final_models, final_predictions = [], [], {}, {}
    with threadpool_limits(limits=2):
        for label, (model_name, count) in test_models.items():
            model = create_model(model_name, None if count == 16 else count)
            model.fit(features_development, labels_development)
            predicted = model.predict(features_test)
            final_models[label], final_predictions[label] = model, predicted
            test_records.append({"model": label, "feature_count": count,
                "accuracy": accuracy_score(labels_test, predicted),
                "macro_f1": f1_score(labels_test, predicted, average="macro"),
                "test_loss": log_loss(labels_test, model.predict_proba(features_test), labels=model.classes_)})
            prediction_tables.append(pd.DataFrame({"model": label, "sample_id": labels_test.index,
                "true_class": labels_test.to_numpy(), "predicted_class": predicted}))
            log_progress(f"Final: {label:12s} | Test_acc: {test_records[-1]['accuracy']:.1%}, "
                         f"Test_loss: {test_records[-1]['test_loss']:.3f}, Test_F1: {test_records[-1]['macro_f1']:.4f}", run_id)
    test_results = pd.DataFrame(test_records)
    save_results(test_results, "test_results.csv", run_id)
    save_results(pd.concat(prediction_tables, ignore_index=True), "test_predictions.csv", run_id)
    print(test_results.round(4))

    figure, axis = plt.subplots(figsize=(7.1, 5.5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(labels_test, final_predictions["LR_selected"],
        labels=class_names, display_labels=class_names, cmap="Blues", colorbar=False,
        xticks_rotation=45, ax=axis, text_kw={"fontsize": 8})
    plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(axis.get_yticklabels(), fontsize=8)
    save_figure(figure, "confusion_matrix.png")

    class_report = pd.DataFrame(classification_report(labels_test, final_predictions["LR_selected"],
        output_dict=True, zero_division=0)).T.loc[class_names]
    save_results(class_report.rename_axis("class_name").reset_index(), "classification_report.csv", run_id)
    print(class_report.round(3))

    # 8. 保存包含标准化步骤的模型
    # 演示模型预先指定为LR_selected，不按测试结果挑选。
    selected_model = final_models["LR_selected"]
    selector = selected_model.named_steps.get("select")
    chosen_features = np.array(feature_names)[selector.get_support()].tolist() if selector is not None else feature_names
    model_file = models_directory / "bean_classifier.joblib"
    joblib.dump({"pipeline": selected_model, "feature_names": feature_names, "selected_features": chosen_features,
            "run_id": run_id, "model_name": "LR_selected", "seed": RANDOM_SEED}, model_file)
    print("实际使用的特征：", chosen_features)
    print("保存模型：", model_file.name)

    # 9. 保存复现信息
    # 保留简单的参数和复现说明；详细指标已经写在CSV中。
    experiment_settings.update({"status": "complete", "validation_fits": len(validation_records) + len(random_records),
        "test_models": len(test_records), "selected_features": chosen_features,
        "jaccard_scores": stability_scores, "elapsed_seconds": time.perf_counter() - started_at,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]},
        "reproduce": "在项目目录执行：.\\.venv\\Scripts\\python.exe train.py；参数保持本文件顶部的固定值。"})
    settings_path.write_text(json.dumps(experiment_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    log_progress(f"本次完成：{experiment_settings['validation_fits']}次验证拟合，6个测试方案，耗时{experiment_settings['elapsed_seconds']:.1f}秒", run_id)
    print("训练完成。接下来打开dry_bean_classification.ipynb，加载本地模型进行预测。")
    log_progress("本次配置：" + json.dumps(experiment_settings, ensure_ascii=False), run_id, show=False)


if __name__ == "__main__":
    main()
