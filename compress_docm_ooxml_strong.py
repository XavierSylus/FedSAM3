import copy
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"

ET.register_namespace("w", W_NS)
ET.register_namespace("m", M_NS)

W = f"{{{W_NS}}}"
M = f"{{{M_NS}}}"


REPLACEMENTS = {
    2: "Abstract. Existing federated brain tumor segmentation often assumes similar modalities, supervision, and update spaces, while real collaboration has missing-modality heterogeneity. FedSAM3-Hetero addresses this setting through server-side restricted routing based on parameter-group compatibility, so clients update only reliable groups and semantically incompatible mixing is reduced. Global representation update is retained as an auxiliary cross-round semantic path. Three protocol-controlled experiments show that multimodal clients improve the image-only baseline in segmentation performance and boundary quality. With the same client composition, restricted routing yields small, consistent gains in boundary quality and training stability, but not a clear Dice gap. FedProx performs better overall, and component analysis finds no stable separable gain from global representation update. FedSAM3-Hetero should therefore be viewed as a framework for analyzing parameter mixing boundaries, not as a universally stronger baseline.",
    5: "Brain tumor segmentation is central to medical image analysis, affecting lesion localization, volume estimation, and treatment planning. Recent pretrained vision and large-scale segmentation models improve anatomical and pathological representation. However, privacy-sensitive medical images are institutionally distributed, making single-center training insufficient. Federated learning enables multi-institutional segmentation without moving local data.",
    6: "Most federated medical segmentation studies assume shared modalities, supervision, and update spaces, so objectives and trainable parameters are consistent. Real collaboration violates this through acquisition, annotation, and information-system differences, creating missing-modality heterogeneity in modality availability, supervision, inputs, and stably updatable parameter subspaces. Uniform FedAvg may mix semantically incompatible updates, causing cross-modal negative transfer and weaker convergence and segmentation performance.",
    7: "The key issue is whether updates from different modality conditions can be directly aggregated, not merely whether modalities are added. Image-only clients mainly optimize segmentation, whereas multimodal clients also follow cross-modal constraints. Indiscriminate aggregation can make functionally different parameters interfere. Thus, constraining parameter mixing boundaries by client update capability is more critical than increasing modality types.",
    8: "FedSAM3-Hetero is a framework for missing-modality heterogeneous federated brain tumor segmentation centered on server-side parameter semantic constraints. Parameter-group admission aligns aggregation with each client's update capability, while cross-round global representations provide semantic anchors. Main experiments include only image-only and multimodal clients; text-only clients remain an extensible setting and do not support the main conclusions.",
    9: "The contributions are threefold. First, the framework unifies client roles, local objectives, and server-side collaboration paths under asymmetric image-only and multimodal participation. Second, restricted routing limits which parameter groups each client type can update, reducing incompatible mixing. Third, Groups A-C, FedProx reference, component ablation, and minimal ablation distinguish effects of multimodal participation and routing constraints on performance and stability.",
    12: "Medical image segmentation localizes organs or lesions. Existing approaches include U-Net-style encoder-decoders with skip connections and extensions, Transformer/hybrid architectures for boundaries and long-range dependencies, and SAM/adapter adaptation as in FedFMS and SAM3-Adapter. These centralized methods address visual representation, boundary recovery, and multi-scale fusion under unified data access and supervision, but not stable collaboration among heterogeneous clients with isolated institutional data.",
    15: "Federated medical image segmentation keeps data local for privacy-preserving multi-center modeling. Prior work includes FedAvg-style averaging, statistical heterogeneity methods for non-IID data, sample imbalance, and distribution shifts, and medical approaches using regularization, structural adaptation, or pretrained transfer. FedFMS incorporates SAM and adapter-based transfer. Most methods still assume homogeneous visual-client update spaces. In contrast, missing-modality clients may have different inputs, local objectives, and stably updatable parameter subspaces, raising whether all parameters should still be averaged.",
    18: "Multimodal federated learning studies distributed collaboration across images, texts, and other modalities. Existing work covers heterogeneous modalities and tasks, missing-modality completion/reconstruction/consistency, and parameter heterogeneity via distillation, module sharing, representation transfer, or partial sharing. CreamFL uses representation-level distillation and contrastive constraints for modality differences and model drift. These studies underexplore parameter responsibility when supervision capabilities differ. This paper asks whether semantic routing constraints can reduce incompatible update mixing while maintaining performance, boundary quality, and stability.",
    19: "Prior studies support centralized segmentation, federated optimization, and multimodal collaboration, but not parameter update aggregability in missing-modality heterogeneous medical segmentation. This paper builds an experimental framework for inconsistent client responsibilities and update capabilities, designs parameter-level routing constraints, and analyzes their stabilizing effect through protocol-controlled validation.",
    21: "FedSAM3-Hetero targets missing-modality heterogeneous federated brain tumor segmentation, where clients differ in modalities, supervision, and stably updatable parameter subspaces. Since uniform averaging may mix incompatible updates, it combines client role partitioning, parameter-group constraints, restricted routing, and global representation update, as summarized in Fig. 1.",
    27: "This paper studies missing-modality heterogeneous federated brain tumor segmentation. Let the client set be [[M0]]. The main experiments include image-only clients with image data and segmentation supervision, and multimodal clients with additional textual semantic information. They are denoted as [[M1]] and [[M2]], respectively, and satisfy Eq. (1):",
    33: "Here, [[M0]], [[M1]], and [[M2]] denote shared, vision-driven, and multimodal semantic-interaction subsets. The goal is parameter-group-compatible collaboration, not uniform averaging, so each client affects only stably updatable subsets.",
    44: "This assumption fails under missing-modality heterogeneity, where clients update different subspaces. FedSAM3-Hetero applies restricted routing: the server partitions parameters by module responsibility, defines eligible clients for each group, and admits only those satisfying the update condition. Table 1 maps modules, groups, and allowed clients.",
    45: "Let S_g denote clients allowed to aggregate the g-th parameter group. Its update after round t is written as Eq. (5):",
    47: "Here, [[M0]] is client [[M1]]'s update for group [[M2]] after round [[M3]], and [[M4]] is the within-group normalized weight. Aggregation is grouped by parameter semantics and admitted by client capability.",
    48: "If no eligible uploader exists, the server keeps the previous value of this parameter group, as shown in Eq. (6):",
    56: "After each round, the server updates global image/text representations from uploaded representations using EMA, as written in Eq. (7):",
    59: "These operations add little overhead: restricted routing only performs whitelist filtering, while global representation update adds one 768-d vector per participating client, about 3 KB per round, or 0.049% to 0.099% of the 5.95 MB trainable upload. This estimate is not effectiveness evidence.",
    62: "FedSAM3-Hetero initializes global parameters and image/text representations. Each round, the server broadcasts parameters; image-only clients optimize segmentation loss, and multimodal clients also optimize cross-modal alignment. The server applies restricted routing to shared, vision, and multimodal groups, updates representations through EMA, evaluates Dice and HD95, and computes gradient conflict for analysis. After [[M0]] rounds, the final global model is obtained.",
    65: "This paper evaluates the framework on a BraTS-style MRI brain tumor segmentation task with image-only and multimodal clients. Image-only clients provide images and segmentation supervision; multimodal clients also include textual semantic information. Text-only clients remain extensible and are excluded from quantitative comparison.",
    66: "Three internal protocols analyze training strategies. Group A is an image-only FedAvg baseline. Group B adds multimodal clients but keeps direct FedAvg. Group C uses the same client composition as Group B and enables restricted routing to evaluate parameter mixing boundaries, boundary quality, and training stability. Group D is an external FedProx robust-optimization reference, not a FedSAM3-Hetero variant.",
    68: "Table 2 compares Groups A-C and external FedProx Group D by client composition, aggregation, restricted routing, FedProx setting, and purpose. Groups A-C examine image-only training, multimodal participation, and parameter-level routing, while Group D prevents interpreting internal changes as general superiority over robust federated methods.",
    69: "All groups use input size 256, 3 classes, batch size 1, 4 accumulation steps, effective batch size 4, learning rate $5×10−5$, and 60 rounds. Current labels are closer to BG, WT, and ET than a full WT/TC/ET loop, so evaluation and visualization focus on WT and ET.",
    70: "Two supplementary analyses clarify design factors: component analysis separates restricted routing from global representation update, and a minimal Group C ablation on $\\lambda_{cream}$ tests whether close B/C metrics stem from insufficient distillation strength. The goal is mechanism analysis, not universal superiority.",
    73: "Table 3 reports Groups A-D under three random seeds. Average gradient conflict angle is used only for B/C mechanism analysis under the same client composition; external FedProx Group D is excluded from this metric.",
    76: "Compared with Group A, Groups B and C improve Best Dice, Final Dice, and Final HD95, showing that multimodal clients improve segmentation performance and boundary quality over the image-only baseline. Text semantics can help without entering the final output space.",
    77: "With the same client composition, Groups B and C keep Dice nearly unchanged, while Group C slightly improves Final HD95 and average gradient conflict angle. Restricted routing therefore mainly constrains parameter mixing boundaries and stabilizes training.",
    78: "FedProx outperforms the internal protocols in Dice and HD95, indicating strong client-drift suppression. Thus, this paper claims multimodal-participation gains and lightweight routing benefits, not overall superiority over classical robust federated methods.",
    83: "Fig. 2 supports interpreting restricted routing as stabilization rather than strong optimization change. B/C gradient conflict angles remain below [[M0]] for most rounds, without uncontrolled conflict. Group C also shows slightly lower conflict angles and smoother HD95, consistent with better boundary quality.",
    84: "This does not prove that restricted routing reshapes optimization or generally beats robust baselines. It only supports limited, directionally consistent stability and boundary-quality improvements under the current setting.",
    88: "To analyze components, this paper compares restricted routing and global representation update, focusing on whether gains mainly come from routing constraints or the auxiliary server-side representation path.",
    89: "Table 3 shows that restricted routing does not clearly separate main metrics under unchanged client composition. Groups B and C remain close in Final Dice, while Group C slightly improves Final HD95 and average gradient conflict angle, indicating boundary and stability effects.",
    92: "Table 4 compares Full C with C w/o global representation update. Removing it does not stably reduce Best Dice, Final Dice, or Final HD95, and some metrics improve slightly. Under the current protocol, data scale, and rounds, this path is auxiliary rather than core evidence.",
    93: "Overall, restricted routing better explains the observations by constraining parameter mixing boundaries and giving limited boundary-quality and stability gains. Global representation update shows no stable independent gain and should remain auxiliary.",
    96: "This paper further tests whether close B/C metrics under Group C are caused by an overly small distillation weight [[M0]]. It adjusts [[M1]] while keeping client composition, routed aggregation, and other training settings unchanged.",
    99: "Table 5 compares Final Dice, Final HD95, and average gradient conflict angle for Group C after 30 rounds under different [[M0]] settings. Increasing [[M1]] from 0.02 to 0.10 and 0.20 gives no monotonic improvement.",
    100: "Under current protocol and data conditions, distillation strength does not dominate the B/C difference. The close metrics should not be attributed simply to an overly small [[M0]] or expected to improve by increasing it. Together with Section 4.4, the evidence points to parameter mixing boundary constraints rather than distillation weight or global representation update.",
    105: "Fig. 3 compares representative validation slices. Cases 0043 and 0044 support restricted routing, where Group C gives slightly more GT-consistent boundary continuity and contour fitting than Group B. Case 0046 is marginal, and case 0049 favors Group B. The improvement is sample-dependent, not universal.",
    108: "Fig. 4 summarizes endpoints: multimodal participation improves over Group A, while restricted routing mainly slightly reduces HD95.",
    110: "Fig. 4. Endpoint summary from Table 3. Bars show Final Dice and Final HD95 with standard-deviation error bars.",
    114: "Limitations remain. First, the main experiments include only image-only and multimodal clients. Text-only clients are extensible but not quantitative, so conclusions mainly apply to coexisting image and image-text clients.",
    115: "Second, the BraTS-style MRI task requires validation on more tasks, client compositions, and multi-center datasets.",
    116: "Third, restricted routing gives limited boundary-quality and stability gains, while global representation update remains auxiliary.",
    117: "Future work may combine restricted routing with FedProx, FedAdam, and larger heterogeneous multi-center settings.",
    119: "This paper proposes FedSAM3-Hetero for missing-modality heterogeneous federated brain tumor segmentation. Its stable core is parameter-level restricted routing, while global representation update remains auxiliary. Groups A-C show that multimodal clients improve the image-only baseline in segmentation performance and boundary quality. Under the same client composition, restricted routing does not substantially enlarge the Dice gap, but gives limited, directionally consistent improvements in Final HD95 and training stability-related metrics. Component analysis and minimal ablation support interpreting the method as a stabilizing decoupled collaboration mechanism focused on parameter mixing boundaries rather than large main-metric gains. Since FedProx performs better overall, the method should not be summarized as comprehensively superior to classical robust federated learning. Future work should validate stability and generalization on more tasks, clients, and multi-center datasets.",
}


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def qname(tag):
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[1]
    return tag


def paragraph_text_and_math(paragraph):
    parts = []
    math_nodes = []
    for child in list(paragraph):
        local = qname(child.tag)
        if local == "r":
            for text_node in child.findall(f".//{W}t"):
                parts.append(text_node.text or "")
        elif local in {"oMath", "oMathPara"}:
            math_nodes.append(copy.deepcopy(child))
            math_text = "".join((t.text or "") for t in child.findall(f".//{M}t"))
            parts.append(f"[[M{len(math_nodes) - 1}:{math_text}]]")
    return "".join(parts), math_nodes


def make_text_run(text):
    run = ET.Element(f"{W}r")
    text_node = ET.SubElement(run, f"{W}t")
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = text
    return run


def replace_paragraph(paragraph, replacement):
    _, math_nodes = paragraph_text_and_math(paragraph)
    paragraph_props = []
    for child in list(paragraph):
        if qname(child.tag) == "pPr":
            paragraph_props.append(copy.deepcopy(child))
            break
    for child in list(paragraph):
        paragraph.remove(child)
    for child in paragraph_props:
        paragraph.append(child)

    pos = 0
    for match in re.finditer(r"\[\[M(\d+)\]\]", replacement):
        if match.start() > pos:
            paragraph.append(make_text_run(replacement[pos:match.start()]))
        math_index = int(match.group(1))
        if math_index >= len(math_nodes):
            raise ValueError(f"Missing math node M{math_index}")
        paragraph.append(copy.deepcopy(math_nodes[math_index]))
        pos = match.end()
    if pos < len(replacement):
        paragraph.append(make_text_run(replacement[pos:]))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compress_docm_ooxml_strong.py INPUT.docm OUTPUT.docm")

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
    touched = []

    for index, replacement in REPLACEMENTS.items():
        paragraph = paragraphs[index - 1]
        old_text, math_nodes = paragraph_text_and_math(paragraph)
        old_words = word_count(old_text)
        new_words = word_count(replacement)
        before_words += old_words
        after_words += new_words
        replace_paragraph(paragraph, replacement)
        touched.append((index, old_words, new_words, len(math_nodes)))

    after_math = len(root.findall(f".//{M}oMath")) + len(root.findall(f".//{M}oMathPara"))
    if before_math != after_math:
        raise RuntimeError(f"Math node count changed: {before_math} -> {after_math}")

    all_entries["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in all_entries.items():
            zout.writestr(name, data)

    print(f"wrote={dst}")
    print(f"math_nodes_before={before_math} math_nodes_after={after_math}")
    print(f"target_words_before={before_words} target_words_after={after_words} reduced={before_words - after_words}")
    for index, old_words, new_words, math_count in touched:
        print(f"P{index}: {old_words} -> {new_words} words, math_nodes={math_count}")


if __name__ == "__main__":
    main()
