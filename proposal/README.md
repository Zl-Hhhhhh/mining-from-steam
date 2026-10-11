# Proposal draft

`Group14_Proposal.tex` is the editable English proposal. It uses the unmodified official NeurIPS 2026 style in non-anonymous preprint mode. The manuscript replaces the conference notice with an AIAA 3111 proposal notice; it is not a NeurIPS submission.

## Fill in before submission

At the top of the LaTeX file, edit `GroupNumber`, `MemberOneName`, `MemberOneID`, `MemberTwoName`, and `MemberTwoID`. There are two member slots, as requested.

The compiled draft is `output/pdf/Group14_Proposal.pdf`. The group number follows the renamed source file; verify that it matches your Canvas group before uploading. Only one member should submit. Both members must join the Canvas group on the People page; this cannot be verified from the repository.

The course reference notice and guide prohibit submitting directly AI-generated work. Both members must read, revise, and take responsibility for this draft before submission. The final report should disclose how AIGC assisted proposal drafting, tooling, or analysis, and should include only references and results the group has verified.

## Reviewed selection

The source `config/games_clustering.json` actually contains 51 entries despite its target of 50. Wallpaper Engine is software and is excluded from the main game study. The reviewed manifest corrects identifier errors while preserving the intended game choices; the original configuration is not modified. `games_reviewed.json` records the source hash, exclusions, corrections, and live store verification. Corrections: Dead Cells is 588650, Dead by Daylight is 381210, and NARAKA: BLADEPOINT is 1203220 (the original 1088850 belongs to Marvel's Guardians of the Galaxy). All 50 retained entries are verified as game applications.

The proposal starts with up to 200 clean English and 200 clean Simplified Chinese reviews per game, for a target of 20,000 across 50 games. It frames the project as Steam data mining across reviews, reviewer activity records, games, and time periods. Topic discovery, clustering, and association analysis are candidate approaches; the pilot will determine the final scope. Structured review fields and store metadata complement the text. News/update records, additional event-window reviews, player-count snapshots, and compatible research datasets are potential supplementary sources, subject to coverage and provenance checks. Full collection and analysis are proposed work; the existing CS2 pilot is not presented as a completed 50-game dataset.

Course deadlines are taken from the references in `refs/`: proposal and group registration October 15, poster November 27, poster session December 4, and final submission December 11, 2026. Check Canvas for any subsequently announced changes.

## Compile on this computer

MiKTeX is installed for the current Windows user. A newly opened terminal should find `pdflatex` on PATH. From the repository root:

```powershell
./.venv/Scripts/python.exe -X utf8 scripts/proposal_tools.py build
./.venv/Scripts/python.exe -X utf8 scripts/proposal_tools.py inspect
```

For direct compilation, open a new PowerShell terminal and run from the repository root:

```powershell
cd proposal
pdflatex --enable-installer -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf Group14_Proposal.tex
pdflatex --enable-installer -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf Group14_Proposal.tex
```

The build helper locates MiKTeX even when an already open terminal has an older PATH. It compiles twice, disables shell escape, and allows MiKTeX to install missing LaTeX packages. Output is placed in `output/pdf/`; auxiliary files and rendered page previews are kept in `tmp/pdfs/`. PDF checks use `pypdf` and `PyMuPDF`, installed in the project virtual environment.

To use Overleaf, upload the `.tex` and `.sty` files together and select `Group14_Proposal.tex` as the main document. No separate BibTeX file is needed. A source ZIP is created by `scripts/proposal_tools.py bundle` in `output/latex/`.

The downloaded style's original source and SHA-256 are in `template-source.json`.
