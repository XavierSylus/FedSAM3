import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from compress_docm_ooxml_strong import (
    M,
    W,
    paragraph_text_and_math,
    qname,
    replace_paragraph,
    word_count,
)


REPLACEMENTS = {
    2: "Abstract. Existing federated brain tumor segmentation often assumes similar modalities, supervision, and update spaces, while real collaboration has missing-modality heterogeneity. FedSAM3-Hetero addresses this setting through server-side restricted routing based on parameter-group compatibility, allowing clients to update only reliable groups and reducing incompatible mixing. Global representation update is retained as an auxiliary semantic path. Three protocol-controlled experiments show that multimodal clients improve the image-only baseline. With the same client composition, restricted routing gives small boundary-quality and stability gains, but no clear Dice gap. FedProx performs better overall, and component analysis finds no stable separable gain from global representation update. FedSAM3-Hetero is therefore a framework for analyzing parameter mixing boundaries, not a universally stronger baseline.",
    5: "Brain tumor segmentation affects lesion localization, volume estimation, and treatment planning. Recent pretrained vision and large-scale segmentation models improve anatomical and pathological representation. However, privacy-sensitive medical images are institutionally distributed, making single-center training insufficient. Federated learning enables multi-institutional segmentation without moving local data.",
    6: "Most federated medical segmentation studies assume shared modalities, supervision, and update spaces, so objectives and trainable parameters are consistent. Real collaboration violates this through acquisition, annotation, and information-system differences, creating missing-modality heterogeneity in modality availability, supervision, inputs, and stably updatable parameter subspaces. Uniform FedAvg may mix incompatible updates, causing cross-modal negative transfer and weaker convergence and segmentation performance.",
    7: "The key issue is whether updates from different modality conditions can be directly aggregated. Image-only clients mainly optimize segmentation, whereas multimodal clients also follow cross-modal constraints. Indiscriminate aggregation can make functionally different parameters interfere, so constraining parameter mixing boundaries by client update capability is more critical than adding modality types.",
    8: "FedSAM3-Hetero is a framework for missing-modality heterogeneous federated brain tumor segmentation centered on server-side parameter semantic constraints. Parameter-group admission aligns aggregation with each client's update capability, while global representations provide semantic anchors. Main experiments include only image-only and multimodal clients; text-only clients remain extensible and do not support the main conclusions.",
    9: "Contributions are threefold. First, the framework unifies client roles, objectives, and server-side collaboration under asymmetric image-only and multimodal participation. Second, restricted routing limits which parameter groups each client type updates, reducing incompatible mixing. Third, Groups A-C, FedProx reference, component ablation, and minimal ablation distinguish effects of multimodal participation and routing constraints.",
    12: "Medical image segmentation localizes organs or lesions. Existing approaches include U-Net-style encoder-decoders, Transformer/hybrid architectures, and SAM/adapter adaptation as in FedFMS and SAM3-Adapter. These centralized methods address visual representation, boundary recovery, and multi-scale fusion under unified data access and supervision, but not stable collaboration among heterogeneous clients.",
    15: "Federated medical image segmentation keeps data local for privacy-preserving multi-center modeling. Prior work includes FedAvg-style averaging, statistical heterogeneity methods for non-IID data, sample imbalance, and shifts, and medical approaches using regularization, structural adaptation, or pretrained transfer. Most methods still assume homogeneous visual-client update spaces, whereas missing-modality clients may have different inputs, objectives, and stably updatable subspaces.",
    18: "Multimodal federated learning studies distributed collaboration across images, texts, and other modalities. Existing work covers heterogeneous tasks, missing-modality completion/reconstruction/consistency, and parameter heterogeneity via distillation, module sharing, transfer, or partial sharing. CreamFL uses representation-level distillation and contrastive constraints. These studies underexplore parameter responsibility when supervision capabilities differ. This paper asks whether semantic routing can reduce incompatible mixing while maintaining performance, boundary quality, and stability.",
    19: "Prior studies support centralized segmentation, federated optimization, and multimodal collaboration, but not parameter update aggregability in missing-modality heterogeneous medical segmentation. This paper builds a framework for inconsistent client responsibilities and update capabilities, designs parameter-level routing constraints, and analyzes their stabilizing effect.",
    21: "FedSAM3-Hetero targets missing-modality heterogeneous federated brain tumor segmentation, where clients differ in modalities, supervision, and stably updatable subspaces. Since uniform averaging may mix incompatible updates, it combines client role partitioning, parameter-group constraints, restricted routing, and global representation update.",
    24: "Fig. 1. FedSAM3-Hetero overview. Group C applies restricted routing, Groups A/B use FedAvg, and the text-only branch is inactive.",
    27: "This paper studies missing-modality heterogeneous federated brain tumor segmentation. Let the client set be [[M0]]. The main experiments include image-only clients and multimodal clients with additional textual semantics, denoted as [[M1]] and [[M2]], satisfying Eq. (1):",
    33: "Here, [[M0]], [[M1]], and [[M2]] denote shared, vision-driven, and multimodal semantic-interaction subsets. The goal is parameter-group-compatible collaboration, so clients affect only stably updatable subsets.",
    36: "Image-only clients use segmentation supervision, while multimodal clients combine segmentation supervision with a cross-modal consistency constraint, as shown in Eq. (3):",
    38: "Here, [[M0]] is segmentation loss, [[M1]] is image-text consistency or alignment, and [[M2]] is a balancing coefficient. Thus, heterogeneous clients differ in objectives and reliable parameter influence.",
    43: "Here, [[M0]] denotes the [[M1]]-th client's parameters after local training, and [[M2]] is the normalized aggregation weight. FedAvg assumes similar update capability over the same parameter space.",
    44: "This fails under missing-modality heterogeneity, where clients update different subspaces. FedSAM3-Hetero applies restricted routing: the server partitions parameters, defines eligible clients for each group, and admits only valid updates. Table 1 maps modules, groups, and clients.",
    47: "Here, [[M0]] is client [[M1]]'s update for group [[M2]] after round [[M3]], and [[M4]] is the normalized weight. Aggregation is grouped by semantics and client capability.",
    50: "This avoids invalid perturbations without valid updates and adapts routing to missing updates across clients and rounds. Table 1 maps components, eligible clients, and server processing.",
    55: "Beyond restricted routing, the server keeps a cross-round global representation path. Parameter aggregation constrains update eligibility, while representation statistics maintain semantic references across rounds.",
    58: "Here, [[M0]] is EMA momentum, and [[M1]] and [[M2]] are image/text representation statistics at round [[M3]]. This path does not determine routing; its contribution is evaluated later.",
    59: "These operations add little overhead: routing performs whitelist filtering, while global representation update adds one 768-d vector per client, about 3 KB per round, or 0.049% to 0.099% of the 5.95 MB trainable upload.",
    62: "FedSAM3-Hetero initializes global parameters and image/text representations. Each round, clients optimize segmentation or segmentation plus cross-modal alignment; the server applies restricted routing, updates representations through EMA, evaluates Dice and HD95, and computes gradient conflict. After [[M0]] rounds, the final model is obtained.",
    65: "The framework is evaluated on a BraTS-style MRI brain tumor segmentation task with image-only and multimodal clients. Text-only clients remain extensible and are excluded from quantitative comparison.",
    66: "Three protocols analyze training strategies: Group A is image-only FedAvg, Group B adds multimodal clients with direct FedAvg, and Group C enables restricted routing under Group B's composition. Group D is an external FedProx reference, not a FedSAM3-Hetero variant.",
    68: "Table 2 compares Groups A-C and FedProx Group D by client composition, aggregation, routing, FedProx setting, and purpose. Groups A-C examine image-only training, multimodal participation, and routing; Group D prevents overinterpreting internal changes.",
    69: "All groups use input size 256, 3 classes, batch size 1, 4 accumulation steps, effective batch size 4, learning rate $5×10−5$, and 60 rounds. Labels are closer to BG, WT, and ET than a full WT/TC/ET loop, so evaluation focuses on WT and ET.",
    70: "Two supplementary analyses clarify design factors: component analysis separates routing from global representation update, and a Group C ablation on $\\lambda_{cream}$ tests whether close B/C metrics stem from insufficient distillation. The goal is mechanism analysis.",
    73: "Table 3 reports Groups A-D under three random seeds. Average gradient conflict angle is used only for B/C mechanism analysis; external FedProx Group D is excluded from this metric.",
    76: "Groups B and C improve Best Dice, Final Dice, and Final HD95 over Group A, showing that multimodal clients improve segmentation performance and boundary quality. Text semantics can help without entering the output space.",
    77: "With the same client composition, Group C keeps Dice close to Group B while slightly improving Final HD95 and average gradient conflict angle. Restricted routing mainly stabilizes training.",
    78: "FedProx outperforms internal protocols in Dice and HD95, indicating strong client-drift suppression. Thus, this paper claims multimodal gains and lightweight routing benefits, not overall superiority over robust federated methods.",
    83: "Fig. 2 supports restricted routing as stabilization rather than strong optimization change. B/C gradient conflict angles remain below [[M0]] for most rounds, and Group C shows smoother HD95, consistent with better boundary quality.",
    84: "This does not prove optimization reshaping or general superiority over robust baselines, only limited stability and boundary-quality improvements under the current setting.",
    88: "Component analysis compares restricted routing and global representation update, focusing on whether gains come mainly from routing constraints or the auxiliary representation path.",
    89: "Table 3 shows that routing does not clearly separate main metrics: Groups B and C remain close in Final Dice, while Group C slightly improves Final HD95 and average gradient conflict angle.",
    92: "Table 4 shows that removing global representation update does not stably reduce Best Dice, Final Dice, or Final HD95. Under the current protocol, data scale, and rounds, this path is auxiliary rather than core evidence.",
    93: "Restricted routing better explains the observations by constraining parameter mixing boundaries and giving limited boundary-quality and stability gains. Global representation update remains auxiliary.",
    96: "This paper tests whether close B/C metrics under Group C are caused by small distillation weight [[M0]], adjusting [[M1]] while keeping client composition, routing, and training settings unchanged.",
    99: "Table 5 compares Final Dice, Final HD95, and average gradient conflict angle for Group C after 30 rounds under different [[M0]]. Increasing [[M1]] from 0.02 to 0.10 and 0.20 gives no monotonic improvement.",
    100: "Distillation strength does not dominate the B/C difference. Close metrics should not be attributed simply to small [[M0]] or expected to improve by increasing it. Section 4.4 points instead to parameter mixing boundary constraints.",
    103: "Overall, multimodal participation improves the image-only baseline, while restricted routing mainly constrains mixing boundaries and yields limited boundary-quality and stability gains.",
    104: "Global representation update and distillation-weight ablation do not separately explain the B/C difference; current evidence supports parameter mixing boundary constraints.",
    105: "Fig. 3 compares validation slices. Cases 0043 and 0044 support restricted routing, case 0046 is marginal, and case 0049 favors Group B, so the improvement is sample-dependent.",
    107: "Fig. 3. Qualitative comparison: A is image-only baseline, B direct FedAvg with multimodal clients, and C restricted routing.",
    111: "Current evidence suggests that the method improves stability and interpretability by constraining parameter mixing boundaries, providing a reusable baseline for future missing-modality studies.",
    114: "Limitations remain. Main experiments include only image-only and multimodal clients; text-only clients are extensible but not quantitative, so conclusions mainly apply to image and image-text clients.",
    119: "FedSAM3-Hetero addresses missing-modality heterogeneous federated brain tumor segmentation. Its stable core is parameter-level restricted routing, while global representation update remains auxiliary. Groups A-C show that multimodal clients improve the image-only baseline. Under the same client composition, routing does not substantially enlarge the Dice gap, but gives limited improvements in Final HD95 and training stability-related metrics. Component analysis and minimal ablation support a stabilizing decoupled-collaboration interpretation focused on parameter mixing boundaries rather than large main-metric gains. Since FedProx performs better overall, the method is not comprehensively superior to classical robust federated learning. Future work should validate stability and generalization on more tasks, clients, and multi-center datasets.",
}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compress_docm_ooxml_onepage.py INPUT.docm OUTPUT.docm")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if src.resolve() == dst.resolve():
        raise SystemExit("Refusing to overwrite input")

    with zipfile.ZipFile(src, "r") as zin:
        all_entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    root = ET.fromstring(all_entries["word/document.xml"])
    body = root.find(f"{W}body")
    paragraphs = [element for element in body if qname(element.tag) == "p"]

    before_math = len(root.findall(f".//{M}oMath")) + len(root.findall(f".//{M}oMathPara"))
    before_words = 0
    after_words = 0
    for index, replacement in REPLACEMENTS.items():
        paragraph = paragraphs[index - 1]
        old_text, _ = paragraph_text_and_math(paragraph)
        before_words += word_count(old_text)
        after_words += word_count(replacement)
        replace_paragraph(paragraph, replacement)

    after_math = len(root.findall(f".//{M}oMath")) + len(root.findall(f".//{M}oMathPara"))
    if before_math != after_math:
        raise RuntimeError(f"Math node count changed: {before_math} -> {after_math}")

    all_entries["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in all_entries.items():
            zout.writestr(name, data)

    print(f"wrote={dst}")
    print(f"math_nodes_before={before_math} math_nodes_after={after_math}")
    print(f"target_words_before={before_words} target_words_after={after_words} reduced={before_words-after_words}")


if __name__ == "__main__":
    main()
