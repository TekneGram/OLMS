Here is a concrete version I would use.

**Definitions**

`clause-like` means the candidate subtree has enough syntax to behave like a clause.

A candidate subtree is **clause-like** if:

1. It has a predicate root:
   - root `UPOS` is `VERB`, or
   - root `UPOS` is `ADJ`/`NOUN` and the subtree contains a `cop` or `aux:pass`.

2. It has a subject/license:
   - contains `nsubj` or `nsubj:pass`, or
   - root has `Mood=Imp`, or
   - dependency label is `conj` and the parent clause has a subject that can be inherited.

3. It has enough substance:
   - at least `4` non-punctuation tokens, and
   - at least `2` content tokens where `UPOS in {NOUN, PROPN, VERB, ADJ, ADV, NUM}`.

`viable` means the split is useful for matching, not just syntactically detectable.

A candidate subtree is **viable** if it is clause-like and also passes these filters:

1. Length:
   - at least `5` total non-punctuation tokens, or
   - at least `3` content tokens.

2. Not just a function/purpose fragment:
   - reject if root has `VerbForm=Inf` and subtree contains `mark=to`, unless root is imperative or the candidate is attached to a recommendation verb.

3. Not just a relative modifier fragment:
   - for `acl:relcl`, require at least `5` non-punctuation tokens and at least one content noun/object/complement besides the relative pronoun.

4. Not parser noise:
   - reject if root `UPOS` is not in `{VERB, ADJ, NOUN}`.
   - reject if the subtree contains no `NOUN`, `PROPN`, or `PRON`.

5. Not too tiny:
   - reject candidates like “to improve clarity,” “which appears,” “because of this,” “that is clear.”

**Updated Rule Set**

```text
1. Split first at sentence boundaries.

2. Inside each sentence, consider only these dependency labels as split candidates:
   conj
   advcl
   acl:relcl

3. For conj:
   split if the child subtree is clause-like.

   clause-like =
     root is VERB, or root is ADJ/NOUN with cop/aux:pass
     AND has nsubj/nsubj:pass OR root Mood=Imp OR can inherit subject from parent
     AND has >= 4 non-punctuation tokens
     AND has >= 2 content tokens

4. For advcl:
   split only if the subtree is viable.

   viable =
     clause-like
     AND has >= 5 non-punctuation tokens OR >= 3 content tokens
     AND is not an infinitival "to" clause
     AND contains at least one NOUN/PROPN/PRON

5. For acl:relcl:
   split only if the subtree is viable, with stricter checks.

   viable relative clause =
     root is VERB or ADJ/NOUN with cop
     AND has nsubj/nsubj:pass
     AND has >= 5 non-punctuation tokens
     AND has >= 3 content tokens
     AND contains a content argument/modifier beyond the relative pronoun
   Otherwise keep it attached to the noun it modifies.

6. Do not split on:
   xcomp
   ccomp
   infinitival advcl marked by "to"
   short relative clauses
   non-clausal modifiers
```

**Practical Thresholds I’d Start With**

Use these defaults:

```text
MIN_NONPUNCT_TOKENS = 5
MIN_CONTENT_TOKENS = 3
CONTENT_UPOS = {NOUN, PROPN, VERB, ADJ, ADV, NUM}
PREDICATE_UPOS = {VERB, ADJ, NOUN}
SPLIT_DEPRELS = {conj, advcl, acl:relcl}
NEVER_SPLIT_DEPRELS = {xcomp, ccomp}
```

One important detail: for `conj`, allow subject inheritance. In your examples, “but does not include a clear reason” has no explicit subject, but it clearly inherits “Your thesis statement.” Without inheritance, you would miss many useful coordinated feedback clauses.
