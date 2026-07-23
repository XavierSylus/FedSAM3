from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from lxml import etree


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
W_NS = NS["w"]


REFERENCE_TEXT = {
    "ref_1": "Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H.: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat. Methods 18, 203-211 (2021).",
    "ref_2": "Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jäger, P.F., Maier-Hein, K.H.: MedNeXt: Transformer-driven scaling of ConvNets for medical image segmentation. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023, LNCS, vol. 14223, pp. 405-415. Springer Nature Switzerland (2023).",
    "ref_3": "Zhang, X., Ou, N., Basaran, B.D., Visentin, M., Qiao, M., Gu, R., Matthews, P.M., Liu, Y., Ye, C., Bai, W.: A foundation model for lesion segmentation on brain MRI with mixture of modality experts. IEEE Trans. Med. Imaging 44(6), 2594-2604 (2025). doi:10.1109/TMI.2025.3540809.",
    "ref_4": "Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J.: SAM-Med3D: A vision foundation model for general-purpose segmentation on volumetric medical images. IEEE Trans. Neural Netw. Learn. Syst. 36(10), 17599-17612 (2025). doi:10.1109/TNNLS.2025.3586694.",
    "ref_5": "Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., Zhou, Y.: TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers. Med. Image Anal. 97, 103280 (2024).",
    "ref_6": "Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. In: IEEE/CVF International Conference on Computer Vision, pp. 4015-4026. IEEE (2023).",
    "ref_7": "Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Segment anything in medical images. Nat. Commun. 15, 654 (2024).",
    "ref_8": "Yan, Z., Song, S., Song, D., Li, Y., Zhou, R., Sun, W., Chen, Z., Kim, S., Ren, H., Liu, T., Li, Q., Li, X., He, L., Sun, L.: SAMed-2: Selective memory enhanced medical Segment Anything Model. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025, LNCS, vol. 15972, pp. 540-550. Springer Nature Switzerland (2025). doi:10.1007/978-3-032-05169-1_52.",
    "ref_9": "Kairouz, P., McMahan, H.B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A.N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., et al.: Advances and open problems in federated learning. Found. Trends Mach. Learn. 14(1-2), 1-210 (2021).",
    "ref_10": "McMahan, H.B., Moore, E., Ramage, D., Hampson, S., y Arcas, B.A.: Communication-efficient learning of deep networks from decentralized data. In: AISTATS 2017, PMLR, vol. 54, pp. 1273-1282 (2017).",
    "ref_11": "Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V.: Federated optimization in heterogeneous networks. In: MLSys 2020 (2020).",
    "ref_12": "Siomos, V., Passerat-Palmbach, J., Tarroni, G.: FedCLAM: Client adaptive momentum with foreground intensity matching for federated medical image segmentation. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025, LNCS, vol. 15965, pp. 247-257. Springer Nature Switzerland (2025). doi:10.1007/978-3-032-04978-0_24.",
    "ref_13": "Shi, Y., Xue, M., Zeng, Y., Zhang, J., Wan, J., Zhou, Y.: FedAMM: Federated learning for brain tumor segmentation with arbitrary missing modalities. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025, LNCS, vol. 15967, pp. 203-213. Springer Nature Switzerland (2025). doi:10.1007/978-3-032-04984-1_20.",
    "ref_14": "Reddi, S.J., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., McMahan, H.B.: Adaptive federated optimization. In: International Conference on Learning Representations (2021).",
    "ref_15": "Liu, Y., Luo, G., Zhu, Y.: FedFMS: Exploring federated foundation models for medical image segmentation. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024, LNCS, vol. 15008, pp. 283-293. Springer Nature Switzerland (2024).",
    "ref_16": "Yu, Q., Liu, Y., Wang, Y., Xu, K., Liu, J.: Multimodal federated learning via contrastive representation ensemble. In: International Conference on Learning Representations (2023).",
    "ref_17": "Xiong, B., Yang, X., Qi, F., Xu, C.: A comprehensive survey on multimodal federated learning: taxonomy, challenges and future directions. ACM Comput. Surv. 56(9), 1-39 (2024).",
    "ref_18": "Le, H.Q., Thwal, C.M., Qiao, Y., Tun, Y.L., Nguyen, M.N.H., Huh, E.-N., Hong, C.S.: Cross-modal prototype based multimodal federated learning under severely missing modality. Inf. Fusion 122, 103219 (2025). doi:10.1016/j.inffus.2025.103219.",
    "ref_19": "Poudel, P., Shrestha, P., Amgain, S., Shrestha, Y.R., Gyawali, P., Bhattarai, B.: CAR-MFL: Cross-modal augmentation by retrieval for multimodal federated learning with missing modalities. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024, LNCS, vol. 15010, pp. 102-112. Springer Nature Switzerland (2024).",
    "ref_20": "Menze, B.H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., Burren, Y., Porz, N., Slotboom, J., Wiest, R., et al.: The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Trans. Med. Imaging 34(10), 1993-2024 (2015).",
    "ref_21": "Correia de Verdier, M., Saluja, R., Gagnon, L., LaBella, D., Baid, U., Tahon, N.E.-H.M., Foltyn-Dumitru, M., Zhang, J., Alafif, M.M., Baig, S., et al.: The 2024 Brain Tumor Segmentation (BraTS) challenge: glioma segmentation on post-treatment MRI. arXiv preprint arXiv:2405.18368 (2024).",
    "ref_22": "Shi, X., Jain, R.K., Li, Y., Hou, R., Cheng, J., Bai, J., Zhao, G., Lin, L., Xu, R., Chen, Y.: TextBraTS: Text-guided volumetric brain tumor segmentation with innovative dataset development and fusion module exploration. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025, LNCS, vol. 15965, pp. 638-648. Springer Nature Switzerland (2025). doi:10.1007/978-3-032-04978-0_61.",
    "ref_23": "Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., Hu, R., Suris Coll-Vinent, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., et al.: SAM 3: Segment Anything with Concepts. In: International Conference on Learning Representations (2026). arXiv preprint arXiv:2511.16719.",
    "ref_24": "Yang, S., Feng, J., Mi, X., Bi, H., Zhang, H., Sun, J.: Improved baselines with synchronized encoding for universal medical image segmentation. In: Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025, LNCS, vol. 15961, pp. 260-270. Springer Nature Switzerland (2025). doi:10.1007/978-3-032-04937-7_25.",
}


TEXT_REPLACEMENTS = {
    "Existing approaches include U-Net-style encoder-decoder networks [1-4], Transformer and hybrid architectures [5], and SAM-based or adapter-based adaptation methods, such as FedFMS and SAM3-Adapter [6-8, 15, 23, 24].": "Existing approaches include modern encoder-decoder and volumetric foundation models [1-4], Transformer and hybrid architectures [5], and SAM-based or adapter-based adaptation methods, including FedFMS and recent SAM-family medical segmentation models [6-8, 15, 23, 24].",
    "Prior work includes FedAvg-style averaging [10], methods for statistical heterogeneity caused by non-IID data, sample imbalance, and distribution shifts [11-14], and medical segmentation methods based on regularization, structural adaptation, or pretrained transfer [15].": "Prior work includes FedAvg-style averaging [10], methods for statistical heterogeneity, client drift, modality imbalance, and distribution shifts [11-14], and medical segmentation methods based on regularization, structural adaptation, or pretrained transfer [15].",
    "Existing work covers heterogeneous tasks, missing-modality completion, reconstruction, consistency learning, and parameter heterogeneity through distillation, module sharing, transfer, or partial sharing [17-19].": "Existing work covers heterogeneous tasks, missing-modality learning, reconstruction, consistency learning, and parameter heterogeneity through distillation, prototype alignment, module sharing, transfer, or partial sharing [17-19].",
}


def paragraph_text(p: etree._Element) -> str:
    return "".join(p.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))


def bookmark_names(p: etree._Element) -> set[str]:
    return {
        node.get(f"{{{W_NS}}}name")
        for node in p.xpath(".//w:bookmarkStart", namespaces=NS)
        if node.get(f"{{{W_NS}}}name")
    }


def replace_plain_text_preserving_markup(p: etree._Element, old: str, new: str) -> bool:
    text_nodes = p.xpath(".//w:t", namespaces=NS)
    if not text_nodes:
        return False
    joined = "".join(node.text or "" for node in text_nodes)
    if old not in joined:
        return False
    updated = joined.replace(old, new)
    text_nodes[0].text = updated
    for node in text_nodes[1:]:
        node.text = ""
    return True


def set_direct_no_numbering(p: etree._Element) -> None:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        ppr = etree.Element(f"{{{W_NS}}}pPr")
        p.insert(0, ppr)
    for num_pr in ppr.findall("w:numPr", NS):
        ppr.remove(num_pr)
    num_pr = etree.Element(f"{{{W_NS}}}numPr")
    num_id = etree.SubElement(num_pr, f"{{{W_NS}}}numId")
    num_id.set(f"{{{W_NS}}}val", "0")
    ppr.append(num_pr)


def rewrite_reference_paragraph(p: etree._Element, ref_name: str, text: str) -> None:
    number = int(ref_name.removeprefix("ref_"))
    kept_children: list[etree._Element] = []
    for child in list(p):
        if child.tag in {f"{{{W_NS}}}bookmarkStart", f"{{{W_NS}}}bookmarkEnd"}:
            kept_children.append(child)
        elif child.xpath(".//w:bookmarkStart | .//w:bookmarkEnd", namespaces=NS):
            kept_children.append(child)
    for child in list(p):
        if child.tag != f"{{{W_NS}}}pPr":
            p.remove(child)
    for child in kept_children:
        p.append(child)
    r = etree.Element(f"{{{W_NS}}}r")
    t = etree.SubElement(r, f"{{{W_NS}}}t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = f"{number}. {text}"
    p.append(r)
    set_direct_no_numbering(p)


def patch_document(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    with zipfile.ZipFile(src, "r") as zin:
        entries = [(info, zin.read(info.filename)) for info in zin.infolist()]
        document_xml = zin.read("word/document.xml")

    parser = etree.XMLParser(remove_blank_text=False, recover=False)
    root = etree.fromstring(document_xml, parser)
    paragraphs = root.xpath("//w:body/w:p", namespaces=NS)

    replacements_done = 0
    for p in paragraphs:
        current = paragraph_text(p)
        for old, new in TEXT_REPLACEMENTS.items():
            if old in current and replace_plain_text_preserving_markup(p, old, new):
                replacements_done += 1
                current = paragraph_text(p)

    if replacements_done != len(TEXT_REPLACEMENTS):
        raise RuntimeError(f"Expected {len(TEXT_REPLACEMENTS)} body text replacements, applied {replacements_done}")

    updated_refs = set()
    for p in paragraphs:
        names = bookmark_names(p)
        for ref_name, text in REFERENCE_TEXT.items():
            if ref_name in names:
                rewrite_reference_paragraph(p, ref_name, text)
                updated_refs.add(ref_name)

    missing = sorted(set(REFERENCE_TEXT) - updated_refs)
    if missing:
        raise RuntimeError(f"Missing reference bookmarks: {missing}")

    updated_xml = etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )

    with zipfile.ZipFile(dst, "w") as zout:
        for info, data in entries:
            if info.filename == "word/document.xml":
                data = updated_xml
            zout.writestr(info, data)

    verify_document(dst)


def verify_document(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        doc_entries = [info for info in zf.infolist() if info.filename == "word/document.xml"]
        if len(doc_entries) != 1:
            raise RuntimeError(f"Expected one word/document.xml, found {len(doc_entries)}")
        root = etree.fromstring(zf.read("word/document.xml"))

    paragraphs = root.xpath("//w:body/w:p", namespaces=NS)
    ref_bookmarks = set()
    for p in paragraphs:
        ref_bookmarks.update(name for name in bookmark_names(p) if name.startswith("ref_"))
    if ref_bookmarks != set(REFERENCE_TEXT):
        raise RuntimeError(f"Reference bookmark mismatch: {sorted(ref_bookmarks ^ set(REFERENCE_TEXT))}")

    instr_text = " ".join(root.xpath("//w:instrText/text()", namespaces=NS))
    linked_refs = set()
    for ref in REFERENCE_TEXT:
        if ref in instr_text:
            linked_refs.add(ref)
    required = {"ref_1", "ref_5", "ref_6", "ref_9", "ref_10", "ref_11", "ref_14", "ref_15", "ref_16", "ref_17", "ref_20", "ref_22"}
    missing_links = sorted(required - linked_refs)
    if missing_links:
        raise RuntimeError(f"Expected linked references missing from citation fields: {missing_links}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    patch_document(args.src, args.dst)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
