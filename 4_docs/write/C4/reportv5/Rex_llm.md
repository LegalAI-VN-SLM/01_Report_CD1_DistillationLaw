  Diagnosis Summary: A2
============================================================
  Total: 2603 samples

  Categories:
    OK      (answer ✓ thinking ✓):  290
    RISKY   (answer ✓ thinking ✗):  898  ← lucky guess
    PARTIAL (answer ✗ thinking ✓):  361  ← reasoning OK, conclusion bad
    WRONG   (answer ✗ thinking ✗): 1054  ← completely wrong

  DPO candidates (RISKY+PARTIAL+WRONG): 2313

  By task type:
    multi_choice   : 91/796 OK (11.4%) | RISKY=269 PARTIAL=108 WRONG=328
    nli            : 165/1128 OK (14.6%) | RISKY=479 PARTIAL=117 WRONG=367
    syllogism      : 34/679 OK (5.0%) | RISKY=150 PARTIAL=136 WRONG=359

-----------

{
  "experiment": "A2",
  "rouge_threshold": 0.3,
  "thinking_threshold": 0.3,
  "total_samples": 2603,
  "dpo_candidates": 2313,
  "categories": {
    "RISKY": 898,
    "OK": 290,
    "WRONG": 1054,
    "PARTIAL": 361
  },
  "by_task_type": {
    "nli": {
      "RISKY": 479,
      "total": 1128,
      "OK": 165,
      "WRONG": 367,
      "PARTIAL": 117
    },
    "multi_choice": {
      "RISKY": 269,
      "total": 796,
      "OK": 91,
      "WRONG": 328,
      "PARTIAL": 108
    },
    "syllogism": {
      "WRONG": 359,
      "total": 679,
      "PARTIAL": 136,
      "RISKY": 150,
      "OK": 34
    }
  }
}