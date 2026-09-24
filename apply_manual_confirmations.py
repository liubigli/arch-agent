"""Fill blank manual decisions with the corresponding automatic verdict."""

import argparse

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook")
    args = parser.parse_args()

    workbook = load_workbook(args.workbook, data_only=False)
    review = workbook["Manual Review"]
    headers = {cell.value: cell.column for cell in review[1]}
    changed = []
    for row in range(2, review.max_row + 1):
        decision = review.cell(row, headers["manual_decision"])
        if decision.value:
            continue
        reliability = review.cell(row, headers["reliability"]).value
        decision.value = reliability
        review.cell(row, headers["manual_note"]).value = (
            "Conferma manuale della valutazione automatica."
        )
        changed.append(
            (
                review.cell(row, headers["question_id"]).value,
                review.cell(row, headers["model"]).value,
                reliability,
            )
        )

    review.data_validations.dataValidation = []
    validation = DataValidation(
        type="list",
        formula1='"correct,ungrounded,unanswered,unverified_no_tool,incerto,needs_review"',
        allow_blank=True,
    )
    validation.errorTitle = "ManualDecisionValidation"
    review.add_data_validation(validation)
    decision_col = get_column_letter(headers["manual_decision"])
    validation.add(f"{decision_col}2:{decision_col}{review.max_row}")

    summary = workbook["Summary"]
    if "A1:H1" in {str(item) for item in summary.merged_cells.ranges}:
        summary.unmerge_cells("A1:H1")
    summary.merge_cells("A1:I1")
    summary["H8"] = "Manual unverified"
    summary["I8"] = "Manual incerto"
    for cell in (summary["H8"], summary["I8"]):
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.font = Font(color="FFFFFF", bold=True)
    model_col = get_column_letter(headers["model"])
    for row in range(9, summary.max_row + 1):
        summary.cell(row, 8).value = (
            f'=COUNTIFS(\'Manual Review\'!{model_col}:{model_col},A{row},'
            f'\'Manual Review\'!{decision_col}:{decision_col},"unverified_no_tool")'
        )
        summary.cell(row, 9).value = (
            f'=COUNTIFS(\'Manual Review\'!{model_col}:{model_col},A{row},'
            f'\'Manual Review\'!{decision_col}:{decision_col},"incerto")'
        )
    summary.column_dimensions["H"].width = 20
    summary.column_dimensions["I"].width = 16

    workbook.save(args.workbook)
    print(f"Updated confirmations: {len(changed)}")
    for item in changed:
        print(item)


if __name__ == "__main__":
    main()
