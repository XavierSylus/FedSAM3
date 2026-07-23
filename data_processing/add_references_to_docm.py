import argparse
import copy
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ET.register_namespace("w", NS["w"])


REFERENCES = [
    "Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F. (eds.) MICCAI 2015, LNCS, vol. 9351, pp. 234-241. Springer, Cham (2015).",
    "Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O.: 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In: Ourselin, S., Joskowicz, L., Sabuncu, M.R., Unal, G., Wells, W. (eds.) MICCAI 2016, LNCS, vol. 9901, pp. 424-432. Springer, Cham (2016).",
    "Milletari, F., Navab, N., Ahmadi, S.A.: V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In: 2016 Fourth International Conference on 3D Vision, pp. 565-571. IEEE (2016).",
    "Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H.: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat. Methods 18, 203-211 (2021).",
    "Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., Zhou, Y.: TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers. Med. Image Anal. 97, 103280 (2024).",
    "Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. In: IEEE/CVF International Conference on Computer Vision, pp. 4015-4026. IEEE (2023).",
    "Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Segment anything in medical images. Nat. Commun. 15, 654 (2024).",
    "Cheng, J., Ye, J., Deng, Z., Chen, J., Li, T., Wang, H., Su, Y., Huang, Z., Chen, J., Jiang, L., Sun, H., He, J., Zhang, S., Zhu, M., Qiao, Y.: SAM-Med2D. arXiv preprint arXiv:2308.16184 (2023).",
    "Kairouz, P., McMahan, H.B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A.N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., et al.: Advances and open problems in federated learning. Found. Trends Mach. Learn. 14(1-2), 1-210 (2021).",
    "McMahan, H.B., Moore, E., Ramage, D., Hampson, S., y Arcas, B.A.: Communication-efficient learning of deep networks from decentralized data. In: AISTATS 2017, PMLR, vol. 54, pp. 1273-1282 (2017).",
    "Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V.: Federated optimization in heterogeneous networks. In: MLSys 2020 (2020).",
    "Karimireddy, S.P., Kale, S., Mohri, M., Reddi, S.J., Stich, S., Suresh, A.T.: SCAFFOLD: Stochastic controlled averaging for federated learning. In: ICML 2020, PMLR, vol. 119, pp. 5132-5143 (2020).",
    "Li, X., Jiang, M., Zhang, X., Kamp, M., Dou, Q.: FedBN: Federated learning on non-IID features via local batch normalization. In: International Conference on Learning Representations (2021).",
    "Reddi, S.J., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., McMahan, H.B.: Adaptive federated optimization. In: International Conference on Learning Representations (2021).",
    "Liu, Y., Luo, G., Zhu, Y.: FedFMS: Exploring federated foundation models for medical image segmentation. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024, LNCS, vol. 15008, pp. 283-293. Springer Nature Switzerland (2024).",
    "Yu, Q., Liu, Y., Wang, Y., Xu, K., Liu, J.: Multimodal federated learning via contrastive representation ensemble. In: International Conference on Learning Representations (2023).",
    "Xiong, B., Yang, X., Qi, F., Xu, C.: A comprehensive survey on multimodal federated learning: taxonomy, challenges and future directions. ACM Comput. Surv. 56(9), 1-39 (2024).",
    "Liu, Y., Chen, Y., Liu, W., Chen, X., Liu, H., Wang, C.: FedMSplit: Correlation-adaptive federated multi-task learning across multimodal split networks. In: ACM MM 2022, pp. 87-96. ACM (2022).",
    "Poudel, P., Shrestha, P., Amgain, S., Shrestha, Y.R., Gyawali, P., Bhattarai, B.: CAR-MFL: Cross-modal augmentation by retrieval for multimodal federated learning with missing modalities. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024, LNCS, vol. 15010, pp. 102-112. Springer Nature Switzerland (2024).",
    "Menze, B.H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., Burren, Y., Porz, N., Slotboom, J., Wiest, R., et al.: The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Trans. Med. Imaging 34(10), 1993-2024 (2015).",
    "Bakas, S., Akbari, H., Sotiras, A., Bilello, M., Rozycki, M., Kirby, J.S., Freymann, J.B., Farahani, K., Davatzikos, C.: Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. Sci. Data 4, 170117 (2017).",
    "Shi, X., Jain, R.K., Li, Y., Hou, R., Cheng, J., Bai, J., Zhao, G., Lin, L., Xu, R., Chen, Y.: TextBraTS: Text-guided volumetric brain tumor segmentation with innovative dataset development and fusion module exploration. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025, LNCS, vol. 15965, pp. 638-648. Springer Nature Switzerland (2025).",
    "Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., Hu, R., Suris Coll-Vinent, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., et al.: SAM 3: Segment Anything with Concepts. In: International Conference on Learning Representations (2026).",
    "Chen, T., Cao, R., Yu, X., Zhu, L., Ding, C., Ji, D., Chen, C., Zhu, Q., Xu, C., Mao, P., Zang, Y.: SAM3-Adapter: Efficient adaptation of Segment Anything 3 for camouflage object segmentation, shadow detection, and medical image segmentation. arXiv preprint arXiv:2511.19425 (2025).",
]


REPLACEMENTS = {
    "Brain tumor segmentation affects lesion localization, volume estimation, and treatment planning. Recent pretrained vision and large-scale segmentation models improve anatomical and pathological representation. However, privacy-sensitive medical images are institutionally distributed, making single-center training insufficient. Federated learning enables multi-institutional segmentation without moving local data.": "Brain tumor segmentation affects lesion localization, volume estimation, and treatment planning [1-4]. Recent pretrained vision and large-scale segmentation models improve anatomical and pathological representation [5-8, 23, 24]. However, privacy-sensitive medical images are institutionally distributed, making single-center training insufficient. Federated learning enables multi-institutional segmentation without moving local data [9].",
    "Most federated medical segmentation studies assume shared modalities, supervision, and update spaces, so objectives and trainable parameters are consistent. Real collaboration violates this through acquisition, annotation, and information-system differences, creating missing-modality heterogeneity in modality availability, supervision, inputs, and stably updatable parameter subspaces. Uniform FedAvg may mix incompatible updates, causing cross-modal negative transfer and weaker convergence and segmentation performance.": "Most federated medical segmentation studies assume shared modalities, supervision, and update spaces, so objectives and trainable parameters are consistent. Real collaboration violates this through acquisition, annotation, and information-system differences, creating missing-modality heterogeneity in modality availability, supervision, inputs, and stably updatable parameter subspaces. Uniform FedAvg [10] may mix incompatible updates, causing cross-modal negative transfer and weaker convergence and segmentation performance.",
    "Contributions are threefold. First, the framework unifies client roles, objectives, and server-side collaboration under asymmetric image-only and multimodal participation. Second, restricted routing limits which parameter groups each client type updates, reducing incompatible mixing. Third, Groups A-C, FedProx reference, component ablation, and minimal ablation distinguish effects of multimodal participation and routing constraints.": "Contributions are threefold. First, the framework unifies client roles, objectives, and server-side collaboration under asymmetric image-only and multimodal participation. Second, restricted routing limits which parameter groups each client type updates, reducing incompatible mixing. Third, Groups A-C, FedProx reference [11], component ablation, and minimal ablation distinguish effects of multimodal participation and routing constraints.",
    "Medical image segmentation localizes organs or lesions. Existing approaches include U-Net-style encoder-decoders, Transformer/hybrid architectures, and SAM/adapter adaptation as in FedFMS and SAM3-Adapter. These centralized methods address visual representation, boundary recovery, and multi-scale fusion under unified data access and supervision, but not stable collaboration among heterogeneous clients.": "Medical image segmentation localizes organs or lesions. Existing approaches include U-Net-style encoder-decoders [1-4], Transformer/hybrid architectures [5], and SAM/adapter adaptation as in FedFMS and SAM3-Adapter [6-8, 15, 23, 24]. These centralized methods address visual representation, boundary recovery, and multi-scale fusion under unified data access and supervision, but not stable collaboration among heterogeneous clients.",
    "Federated medical image segmentation keeps data local for privacy-preserving multi-center modeling. Prior work includes FedAvg-style averaging, statistical heterogeneity methods for non-IID data, sample imbalance, and shifts, and medical approaches using regularization, structural adaptation, or pretrained transfer. Most methods still assume homogeneous visual-client update spaces, whereas missing-modality clients may have different inputs, objectives, and stably updatable subspaces.": "Federated medical image segmentation keeps data local for privacy-preserving multi-center modeling [9]. Prior work includes FedAvg-style averaging [10], statistical heterogeneity methods for non-IID data, sample imbalance, and shifts [11-14], and medical approaches using regularization, structural adaptation, or pretrained transfer [15]. Most methods still assume homogeneous visual-client update spaces, whereas missing-modality clients may have different inputs, objectives, and stably updatable subspaces.",
    "Multimodal federated learning studies distributed collaboration across images, texts, and other modalities. Existing work covers heterogeneous tasks, missing-modality completion/reconstruction/consistency, and parameter heterogeneity via distillation, module sharing, transfer, or partial sharing. CreamFL uses representation-level distillation and contrastive constraints. These studies underexplore parameter responsibility when supervision capabilities differ. This paper asks whether semantic routing can reduce incompatible mixing while maintaining performance, boundary quality, and stability.": "Multimodal federated learning studies distributed collaboration across images, texts, and other modalities [16, 17]. Existing work covers heterogeneous tasks, missing-modality completion/reconstruction/consistency, and parameter heterogeneity via distillation, module sharing, transfer, or partial sharing [17-19]. CreamFL uses representation-level distillation and contrastive constraints [16]. These studies underexplore parameter responsibility when supervision capabilities differ. This paper asks whether semantic routing can reduce incompatible mixing while maintaining performance, boundary quality, and stability.",
    "The framework is evaluated on a BraTS-style MRI brain tumor segmentation task with image-only and multimodal clients. Text-only clients remain extensible and are excluded from quantitative comparison.": "The framework is evaluated on a BraTS-style MRI brain tumor segmentation task with image-only and multimodal clients [20, 21]. Text-only clients remain extensible and are excluded from quantitative comparison; textual semantics follow the motivation of TextBraTS-style settings [22].",
    "Three protocols analyze training strategies: Group A is image-only FedAvg, Group B adds multimodal clients with direct FedAvg, and Group C enables restricted routing under Group B's composition. Group D is an external FedProx reference, not a FedSAM3-Hetero variant.": "Three protocols analyze training strategies: Group A is image-only FedAvg [10], Group B adds multimodal clients with direct FedAvg, and Group C enables restricted routing under Group B's composition. Group D is an external FedProx reference [11], not a FedSAM3-Hetero variant.",
    "Future work may combine restricted routing with FedProx, FedAdam, and larger heterogeneous multi-center settings.": "Future work may combine restricted routing with FedProx [11], FedAdam [14], and larger heterogeneous multi-center settings.",
}


def get_text(paragraph: ET.Element) -> str:
    return "".join(t.text or "" for t in paragraph.findall(".//w:t", NS))


def get_style_id(paragraph: ET.Element) -> str | None:
    p_pr = paragraph.find("w:pPr", NS)
    if p_pr is None:
        return None
    p_style = p_pr.find("w:pStyle", NS)
    if p_style is None:
        return None
    return p_style.get(W + "val")


def set_paragraph_text(paragraph: ET.Element, new_text: str) -> None:
    text_nodes = paragraph.findall(".//w:t", NS)
    if not text_nodes:
        run = paragraph.find("w:r", NS)
        if run is None:
            run = ET.SubElement(paragraph, W + "r")
        t = ET.SubElement(run, W + "t")
        t.text = new_text
        return
    text_nodes[0].text = new_text
    for node in text_nodes[1:]:
        node.text = ""


def replace_text_paragraphs(root: ET.Element) -> list[str]:
    pending = dict(REPLACEMENTS)
    for paragraph in root.findall(".//w:p", NS):
        current = re.sub(r"\s+", " ", get_text(paragraph)).strip()
        if current in pending:
            set_paragraph_text(paragraph, pending.pop(current))
    return list(pending.keys())


def replace_references(root: ET.Element) -> None:
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Missing document body")
    children = list(body)
    ref_heading_idx = None
    for idx, child in enumerate(children):
        if child.tag == W + "p" and re.sub(r"\s+", " ", get_text(child)).strip() == "References":
            ref_heading_idx = idx
            break
    if ref_heading_idx is None:
        raise RuntimeError("References heading not found")

    remove_indices = []
    template_paragraph = None
    for idx in range(ref_heading_idx + 1, len(children)):
        child = children[idx]
        if child.tag == W + "sectPr":
            break
        if child.tag == W + "p" and get_style_id(child) == "33":
            if template_paragraph is None:
                template_paragraph = child
            remove_indices.append(idx)
        elif child.tag == W + "p" and not get_text(child).strip():
            remove_indices.append(idx)
        else:
            break
    if template_paragraph is None:
        raise RuntimeError("Reference item template paragraph not found")

    for idx in reversed(remove_indices):
        body.remove(children[idx])

    insert_at = ref_heading_idx + 1
    for offset, ref in enumerate(REFERENCES):
        new_p = copy.deepcopy(template_paragraph)
        set_paragraph_text(new_p, ref)
        body.insert(insert_at + offset, new_p)


def update_app_props(docx_dir: Path) -> None:
    app_path = docx_dir / "docProps" / "app.xml"
    if not app_path.exists():
        return
    tree = ET.parse(app_path)
    root = tree.getroot()
    for child in root:
        tag = child.tag.split("}", 1)[-1]
        if tag == "Words":
            child.text = ""
        elif tag == "Pages":
            child.text = ""
    tree.write(app_path, encoding="utf-8", xml_declaration=True)


def patch_docm(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(input_path, "r") as zin:
            zin.extractall(tmp_dir)

        document_path = tmp_dir / "word" / "document.xml"
        tree = ET.parse(document_path)
        root = tree.getroot()
        missing = replace_text_paragraphs(root)
        if missing:
            raise RuntimeError("Text replacements not found: " + "; ".join(missing[:3]))
        replace_references(root)
        tree.write(document_path, encoding="utf-8", xml_declaration=True)
        update_app_props(tmp_dir)

        if output_path.exists():
            output_path.unlink()
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for file_path in tmp_dir.rglob("*"):
                if file_path.is_file():
                    zout.write(file_path, file_path.relative_to(tmp_dir).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    patch_docm(args.input, args.output)


if __name__ == "__main__":
    main()
