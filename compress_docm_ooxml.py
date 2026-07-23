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
    2: (
        "Abstract. Existing federated brain tumor segmentation methods usually assume similar modalities, supervision, and parameter update spaces across clients, whereas real collaboration often has missing-modality heterogeneity. This paper proposes FedSAM3-Hetero for this setting. It introduces server-side restricted routing based on parameter-group compatibility, allowing clients to update only reliable groups and reducing semantically incompatible mixing. Global representation update is retained as an auxiliary cross-round semantic path. Three protocol-controlled experiments show that multimodal clients improve segmentation performance and boundary quality over the image-only baseline. With the same client composition, restricted routing does not clearly enlarge the Dice gap between Groups B and C, but yields small, consistent gains in boundary quality and training stability. A supplementary FedProx baseline performs better overall, indicating strong competition from client drift suppression. Component analysis shows no stable separable gain from global representation update. Overall, FedSAM3-Hetero analyzes parameter mixing boundaries and stabilizes heterogeneous collaboration rather than universally outperforming strong baselines."
    ),
    5: (
        "Brain tumor segmentation is central to medical image analysis, affecting lesion localization, volume estimation, and treatment planning. Recent pretrained vision and large-scale segmentation models improve complex anatomical and pathological representation. However, privacy-sensitive medical images are distributed across institutions, making single-center training insufficient. Federated learning enables multi-institutional segmentation without moving local data."
    ),
    6: (
        "Most federated medical segmentation studies assume clients share modalities, supervision, and update spaces, so objectives and trainable parameters are consistent. In real collaboration, acquisition protocols, annotations, and information systems create missing-modality heterogeneity: institutions differ in modality availability, supervision, input sources, and stably updatable parameter subspaces. Uniform FedAvg may mix semantically incompatible updates, causing cross-modal negative transfer and weakening convergence stability and segmentation performance."
    ),
    7: (
        "In this setting, the key issue is whether updates from different modality conditions can be directly aggregated, not merely adding modalities. Image-only clients mainly optimize segmentation, whereas multimodal clients also follow cross-modal constraints, producing explicit cross-modal semantics. Indiscriminate aggregation can make parameters with different functions interfere. Thus, constraining parameter mixing boundaries by client update capability is more critical than increasing modality types."
    ),
    8: (
        "This paper proposes FedSAM3-Hetero, a framework for missing-modality heterogeneous federated brain tumor segmentation centered on server-side parameter semantic constraints. Parameter-group admission conditions align aggregation with each client's update capability, while cross-round global representations provide semantic anchors. The main experiments include only image-only and multimodal clients; text-only clients remain an extensible framework-level setting and do not directly support the main conclusions."
    ),
    9: (
        "Contributions are threefold. First, the framework unifies client roles, local objectives, and server-side collaboration paths under asymmetric image-only and multimodal participation. Second, restricted routing based on parameter semantic compatibility limits which groups each client type can update, reducing incompatible mixing. Third, Groups A-C, an external FedProx reference, component ablation, and minimal ablation distinguish effects of multimodal participation, routing constraints, and related components on performance and stability."
    ),
    12: (
        "Medical image segmentation localizes organs or lesions in medical images. Existing approaches include U-Net-style encoder-decoders with skip connections and extensions such as volumetric modeling, automatic configuration, and task adaptation; Transformer/hybrid architectures for boundaries and long-range dependencies; and SAM/adapter foundation model adaptation, as in FedFMS and SAM3-Adapter. Effective in centralized segmentation, these methods address visual representation, boundary recovery, and multi-scale fusion under unified data access and supervision, but do not explain stable collaboration among heterogeneous clients with isolated institutional data."
    ),
    15: (
        "Federated medical image segmentation enables privacy-preserving multi-center modeling by keeping data local. Prior work includes FedAvg-style parameter averaging; statistical heterogeneity methods for non-IID data, sample imbalance, and local distribution shifts; and multi-center medical approaches using regularization, structural adaptation, or pretrained transfer. FedFMS incorporates SAM and adapter-based foundation model transfer. However, most methods still assume homogeneous update spaces across visual clients. The missing-modality setting differs because clients may have different inputs, local objectives, and stably updatable parameter subspaces, raising whether the server should still average all parameters."
    ),
    18: (
        "Multimodal federated learning studies distributed collaboration across images, texts, and other modalities. Existing work covers heterogeneous modalities and tasks, missing-modality settings handled by completion, reconstruction, or consistency constraints, and parameter heterogeneity addressed through knowledge distillation, module sharing, representation transfer, or partial sharing. CreamFL uses representation-level distillation and contrastive constraints for modality differences and model drift. These studies consider modality missingness, structural heterogeneity, and knowledge transfer, but underexplore parameter responsibility when supervision capabilities differ. This paper focuses on whether semantic routing constraints can reduce incompatible update mixing while maintaining performance, boundary quality, and stability."
    ),
    19: (
        "Overall, prior studies support centralized segmentation, federated optimization, and multimodal collaboration, but have not systematically answered parameter update aggregability in missing-modality heterogeneous medical segmentation. This paper builds an experimental framework for inconsistent client responsibilities and update capabilities, designs parameter-level routing constraints, and analyzes their stabilizing effect through protocol-controlled validation."
    ),
    21: (
        "FedSAM3-Hetero addresses missing-modality heterogeneous federated brain tumor segmentation, where clients differ in input modalities, supervision, and stably updatable parameter subspaces. Since uniform averaging may mix semantically incompatible updates, it combines client role partitioning, parameter-group constraints, restricted routing, and global representation update, as summarized in Fig. 1."
    ),
    27: (
        "This paper studies missing-modality heterogeneous federated brain tumor segmentation. Let the client set be [[M0]]. The main experiments include image-only clients with image data and segmentation supervision, and multimodal clients with image data, segmentation supervision, and textual semantic information. They are denoted as [[M1]] and [[M2]], respectively, and satisfy Eq. (1):"
    ),
    33: (
        "Here, [[M0]], [[M1]], and [[M2]] denote shared, vision-driven, and multimodal semantic-interaction subsets. The objective is not uniform averaging, but collaboration constrained by parameter-group compatibility so each client affects only stably updatable subsets."
    ),
    44: (
        "This assumption fails under missing-modality heterogeneity, where clients stably update different parameter subspaces. FedSAM3-Hetero applies restricted routing based on parameter-group compatibility: the server partitions parameters by module responsibility, defines eligible clients for each group, and admits only those satisfying the update condition. Table 1 maps modules, parameter groups, and allowed clients."
    ),
    45: (
        "Let S_g denote clients allowed to aggregate the g-th parameter group. Its update after communication round t is written as Eq. (5):"
    ),
    47: (
        "Here, [[M0]] is client [[M1]]'s local update for group [[M2]] after round [[M3]], and [[M4]] is the within-group normalized weight. Aggregation is grouped by parameter semantics, admitted by client capability, and executed independently for each group."
    ),
    48: (
        "If no eligible uploader exists in the current round, the server retains the previous global value of this parameter group, as shown in Eq. (6):"
    ),
    56: (
        "After each communication round, the server updates global image and text representations from client-uploaded representations using EMA, as written in Eq. (7):"
    ),
    59: (
        "These server-side operations add little overhead: restricted routing only performs whitelist filtering without extra client communication, while global representation update adds one 768-d vector per participating client, about 3 KB per round, or 0.049% to 0.099% of the 5.95 MB trainable parameter upload. This estimate is not evidence of global representation update effectiveness."
    ),
    62: (
        "FedSAM3-Hetero initializes global model parameters and image/text representations. In each round, the server broadcasts parameters; image-only clients optimize segmentation loss, while multimodal clients jointly optimize segmentation loss and cross-modal alignment. The server applies restricted routing to shared, vision, and multimodal parameter groups, then updates image/text representations through EMA. Dice and HD95 are evaluated on the validation set, and gradient conflict is computed for analysis. After [[M0]] communication rounds, the final global model is obtained."
    ),
    65: (
        "This paper evaluates the framework on a BraTS-style MRI brain tumor segmentation task with image-only and multimodal clients. Image-only clients provide image inputs and segmentation supervision; multimodal clients also include textual semantic information. Text-only clients remain an extensible setting and are excluded from quantitative comparison."
    ),
    66: (
        "To analyze training strategies, this paper constructs three internal protocol-controlled settings. Group A is an image-only FedAvg baseline. Group B adds multimodal clients while retaining direct FedAvg to observe heterogeneous multimodal collaboration. Group C uses the same client composition as Group B but enables restricted routing to evaluate effects on parameter mixing boundaries, boundary quality, and training stability. Group D is an external FedProx baseline, not a FedSAM3-Hetero variant or direct routing test, and provides a strong robust optimization reference."
    ),
    68: (
        "Table 2 compares Groups A, B, and C with the external FedProx baseline, Group D, by client composition, aggregation method, restricted routing, FedProx setting, and purpose. Groups A-C examine image-only federated training, multimodal participation, and parameter-level restricted routing, while Group D prevents internal protocol changes from being interpreted as general superiority over robust federated methods."
    ),
    70: (
        "Two supplementary analyses clarify design factors. Component analysis separates restricted routing from server-side global representation update. A minimal ablation on $\\lambda_{cream}$ within Group C tests whether the close B/C metrics stem from insufficient distillation strength. These settings analyze multimodal participation, parameter mixing boundaries, and training stability rather than universal superiority."
    ),
    73: (
        "Table 3 reports results for Groups A-D under three random seeds. The average gradient conflict angle is used only for B/C mechanism analysis under the same client composition; Group D is an external FedProx baseline and is excluded from this metric."
    ),
    76: (
        "Compared with Group A, Groups B and C achieve better Best Dice, Final Dice, and Final HD95, showing that multimodal clients improve segmentation performance and boundary quality over the image-only baseline. Textual semantic information can therefore provide useful collaborative signals without directly entering the final output space."
    ),
    77: (
        "With the same client composition, Groups B and C show almost unchanged Dice, while Group C slightly and consistently improves Final HD95 and average gradient conflict angle. Thus, restricted routing mainly constrains parameter mixing boundaries and stabilizes training rather than substantially raising the main segmentation metric."
    ),
    78: (
        "The external FedProx baseline outperforms the internal protocol-controlled settings in Dice and HD95, indicating strong competition from client drift suppression. Therefore, this paper does not claim overall superiority over classical robust federated methods; it only supports multimodal participation and lightweight boundary-quality and stability benefits from restricted routing."
    ),
    83: (
        "As shown in Fig. 2, training dynamics support interpreting restricted routing as a stabilizing mechanism, not a strong optimization-changing intervention. The gradient conflict angles of Groups B and C remain below [[M0]] for most rounds, so heterogeneous multimodal collaboration does not enter uncontrolled conflict. Group C also shows slightly lower conflict angles and a smoother HD95 trajectory, consistent with better boundary quality."
    ),
    88: (
        "To analyze component roles, this paper compares restricted routing and global representation update. The focus is not a full factorial decomposition of all modules, but whether current gains mainly come from routing constraints or the auxiliary server-side representation path."
    ),
    84: (
        "This phenomenon does not show that restricted routing reshapes overall optimization or has a general advantage over strong robust baselines. A cautious interpretation is limited but directionally consistent improvement in training stability and boundary quality under the current setting."
    ),
    89: (
        "Table 3 shows that, with unchanged client composition, restricted routing does not clearly separate the main task metrics. Groups B and C remain close in Final Dice, while Group C slightly and consistently improves Final HD95 and average gradient conflict angle. Thus, restricted routing mainly constrains parameter mixing boundaries and stabilizes training behavior and boundary quality."
    ),
    92: (
        "Table 4 compares Full C with C w/o global representation update. Removing global representation update does not stably reduce Best Dice, Final Dice, or Final HD95, and some metrics even improve slightly. Under the current protocol, data scale, and training rounds, this path shows no clear, separable, stable additional gain and should be treated as auxiliary rather than core evidence."
    ),
    93: (
        "Overall, restricted routing better explains the current observations by constraining parameter mixing boundaries and providing limited but consistent boundary-quality and stability gains. In contrast, global representation update shows no stable independent gain and should be described as auxiliary, not as a core contribution parallel to restricted routing."
    ),
    96: (
        "After the component analysis, this paper examines whether the close B/C metrics under Group C are mainly caused by an overly small distillation weight [[M0]]. It therefore performs a minimal within-group adjustment of [[M1]] while keeping client composition, routed aggregation, and all other training settings unchanged."
    ),
    99: (
        "Table 5 compares Final Dice, Final HD95, and average gradient conflict angle for Group C after 30 rounds under different [[M0]] settings. Increasing [[M1]] from 0.02 to 0.10 and 0.20 does not yield monotonic improvement, so simply increasing distillation weight is not stable under current conditions."
    ),
    100: (
        "Under the current protocol and data conditions, distillation strength does not dominate the B/C difference. The close main metrics should not be attributed simply to an overly small [[M0]], nor imply that Group C would gain more as distillation weight increases. Together with Section 4.4, the evidence points to parameter mixing boundary constraints rather than distillation weight or server-side global representation update."
    ),
    105: (
        "Beyond quantitative metrics, Fig. 3 compares representative validation slices. Cases 0043 and 0044 support restricted routing because Group C gives slightly more GT-consistent boundary continuity and contour fitting than Group B. Case 0046 is marginal, and case 0049 is a failure case where Group B is closer to GT. Thus, the boundary improvement is sample-dependent and reflects limited stabilization, not universal gain."
    ),
    108: (
        "Fig. 4 summarizes endpoints: multimodal participation improves over Group A, while restricted routing mainly slightly reduces HD95 under the same client composition."
    ),
    114: (
        "Although this paper constructs a unified framework and analyzes multimodal participation and restricted routing through Groups A-C, limitations remain. First, the main experiments include only image-only and multimodal clients. Text-only clients are only an extensible framework-level setting, not used in quantitative experiments; therefore, conclusions mainly apply to coexisting image and image-text clients."
    ),
    115: (
        "Second, the experiments use a BraTS-style MRI brain tumor segmentation task, requiring validation on additional tasks, more complex client compositions, and larger multi-center datasets."
    ),
    116: (
        "Third, restricted routing yields limited boundary-quality and stability improvements rather than clear main Dice separation. Global representation update remains auxiliary under the current protocol."
    ),
    119: (
        "This paper proposes FedSAM3-Hetero for missing-modality heterogeneous federated brain tumor segmentation. Its stable core is parameter-level restricted routing, which constrains parameter mixing boundaries, while global representation update remains auxiliary. Groups A-C show that multimodal clients improve segmentation performance and boundary quality over the image-only baseline. Comparing Groups B and C under the same client composition shows that restricted routing does not substantially enlarge the Dice gap, but yields limited, directionally consistent improvements in Final HD95 and training stability-related metrics. Component analysis and minimal ablation support interpreting the method as a stabilizing decoupled collaboration mechanism focused on parameter mixing boundaries rather than large main-metric gains. Since FedProx performs better overall, the method should not be summarized as comprehensively superior to classical robust federated learning. Future work may combine restricted routing with stronger federated optimization and validate stability and generalization on more tasks, clients, and multi-center datasets."
    ),
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
            for t in child.findall(f".//{W}t"):
                parts.append(t.text or "")
        elif local in {"oMath", "oMathPara"}:
            math_nodes.append(copy.deepcopy(child))
            mtxt = "".join((t.text or "") for t in child.findall(f".//{M}t"))
            parts.append(f"[[M{len(math_nodes)-1}:{mtxt}]]")
    return "".join(parts), math_nodes


def make_text_run(text):
    run = ET.Element(f"{W}r")
    t = ET.SubElement(run, f"{W}t")
    if text[:1].isspace() or text[-1:].isspace():
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return run


def replace_paragraph(paragraph, replacement):
    _, math_nodes = paragraph_text_and_math(paragraph)
    preserved = []
    for child in list(paragraph):
        if qname(child.tag) == "pPr":
            preserved.append(copy.deepcopy(child))
            break
    for child in list(paragraph):
        paragraph.remove(child)
    for child in preserved:
        paragraph.append(child)

    pos = 0
    for match in re.finditer(r"\[\[M(\d+)\]\]", replacement):
        if match.start() > pos:
            paragraph.append(make_text_run(replacement[pos:match.start()]))
        idx = int(match.group(1))
        if idx >= len(math_nodes):
            raise ValueError(f"Replacement requests math node M{idx}, but only {len(math_nodes)} exist")
        paragraph.append(copy.deepcopy(math_nodes[idx]))
        pos = match.end()
    if pos < len(replacement):
        paragraph.append(make_text_run(replacement[pos:]))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compress_docm_ooxml.py INPUT.docm OUTPUT.docm")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if src.resolve() == dst.resolve():
        raise SystemExit("Refusing to overwrite input")

    with zipfile.ZipFile(src, "r") as zin:
        document_xml = zin.read("word/document.xml")
        all_entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    root = ET.fromstring(document_xml)
    body = root.find(f"{W}body")
    paragraphs = [el for el in body if qname(el.tag) == "p"]

    before_words = 0
    after_words = 0
    before_math = len(root.findall(f".//{M}oMath")) + len(root.findall(f".//{M}oMathPara"))
    touched = []
    for index, replacement in REPLACEMENTS.items():
        paragraph = paragraphs[index - 1]
        old_text, math_nodes = paragraph_text_and_math(paragraph)
        before_words += word_count(old_text)
        after_words += word_count(replacement)
        replace_paragraph(paragraph, replacement)
        touched.append((index, word_count(old_text), word_count(replacement), len(math_nodes)))

    after_math = len(root.findall(f".//{M}oMath")) + len(root.findall(f".//{M}oMathPara"))
    if before_math != after_math:
        raise RuntimeError(f"Math node count changed: before={before_math}, after={after_math}")

    new_document = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    all_entries["word/document.xml"] = new_document

    dst.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(dst, "w", compression=compression) as zout:
        for name, data in all_entries.items():
            zout.writestr(name, data)

    print(f"wrote={dst}")
    print(f"math_nodes_before={before_math} math_nodes_after={after_math}")
    print(f"target_words_before={before_words} target_words_after={after_words} reduced={before_words-after_words}")
    for item in touched:
        print(f"P{item[0]}: {item[1]} -> {item[2]} words, math_nodes={item[3]}")


if __name__ == "__main__":
    main()
