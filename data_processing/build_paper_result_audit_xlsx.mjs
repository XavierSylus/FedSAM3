import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const configPath = new URL("./paper_result_audit_config.json", import.meta.url);
const config = JSON.parse(await fs.readFile(configPath, "utf8"));
const outputDir = config.output_dir;

function parseCsv(text) {
  const rows = [];
  let cell = "";
  let row = [];
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r" && ch !== "\ufeff") {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

async function readRows(fileName) {
  const text = await fs.readFile(path.join(outputDir, fileName), "utf8");
  return parseCsv(text);
}

function writeMatrix(sheet, startCell, values) {
  const range = sheet.getRange(startCell).resize(values.length, values[0].length);
  range.values = values;
}

const workbook = Workbook.create();
const audit = workbook.worksheets.add("Audit Summary");
const seed = workbook.worksheets.add("Seed Values");
const checks = workbook.worksheets.add("Raw Checks");

writeMatrix(audit, "A1", await readRows("paper_result_traceability_audit.csv"));
writeMatrix(seed, "A1", await readRows("seed_level_raw_values.csv"));
writeMatrix(checks, "A1", await readRows("raw_consistency_checks.csv"));

for (const sheet of [audit, seed, checks]) {
  sheet.getRange("A1:Z1").format.font.bold = true;
  sheet.freezePanes.freezeRows(1);
  sheet.getUsedRange().format.wrapText = true;
}

audit.getRange("A:A").format.columnWidthPx = 170;
audit.getRange("B:B").format.columnWidthPx = 130;
audit.getRange("C:C").format.columnWidthPx = 135;
audit.getRange("D:E").format.columnWidthPx = 135;
audit.getRange("L:L").format.columnWidthPx = 620;
audit.getRange("M:N").format.columnWidthPx = 110;
seed.getRange("A:M").format.columnWidthPx = 145;
seed.getRange("K:M").format.columnWidthPx = 430;
checks.getRange("A:H").format.columnWidthPx = 150;
checks.getRange("G:G").format.columnWidthPx = 520;

const output = await SpreadsheetFile.exportXlsx(workbook);
const xlsxPath = path.join(outputDir, "paper_result_traceability_audit.xlsx");
await output.save(xlsxPath);
console.log(xlsxPath);
