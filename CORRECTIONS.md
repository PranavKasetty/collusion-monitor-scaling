# Corrections

## 20 Nov 2026: the vote-count comparison is withdrawn

**What the submitted report says.** The abstract ("Simply counting votes for the colluders' preferred option beat every AI
monitor"), section 4.4 ("Counting votes beats every LLM monitor") and the discussion ("no LLM monitor beats a vote count
that knows the target option") present the stance baseline as outperforming the LLM monitors. This README said the same.

**Why it is wrong.** The stance baseline scores each agent by whether its final recommendation names the colluders' target
option. It is told the target; the monitors are not. The report states this in its method and limitations sections, but the
headline comparison still treats the two as competitors. With unequal information it is not a fair comparison, so the claim
that the vote count "beats" the monitors is withdrawn. The stance baseline remains an upper reference, as the limitations
section already says, and it would also flag lone advocates as colluders.

**What stands.** The other findings do not depend on this comparison: the inverted monitor with the standard prompt, the
shared-wording cause and the one-sentence fix for GLM-5.3-flash (not Gemma 4 26B), the lone-advocate result, and the low
catch rates at a 5% false-flag budget (Table 3).

**What changed.** README findings list updated. The submitted report (`report/`) is left as submitted to Apart Research;
read it with this correction. The project page at pranavkasetty.github.io carries the same correction.
