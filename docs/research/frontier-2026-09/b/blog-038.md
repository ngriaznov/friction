# Fifty Backend Interviews in One Quarter: What Actually Predicted Anything

Last quarter I was the hiring manager for a senior backend role, and between me and my panel we ran fifty full technical loops. That's enough to notice patterns, and enough to be embarrassed about some of the questions I had been asking with total confidence for years.

Here's what I learned, sorted roughly from "I'd bet on this" to "I was wrong."

## What turned out to be strong signal

**Debugging a real-ish system with a real-ish bug.** We gave candidates a small service, three files, with an intermittent failure under load. No trick, just a race condition in a cache refresh. The candidates who did well on this were the ones our team later rated highest in the first ninety days. What it measured wasn't cleverness; it was whether someone forms hypotheses, checks them, and narrates what they're ruling out. People who were quiet for ten minutes and then said "it's the cache" often got the right answer, but they were harder to work with in the pair sessions, and that carried into the job.

**"Tell me about a time you were wrong about a technical decision."** I expected this to be a soft question. It wasn't. Strong candidates had a specific story with a real cost and a clear thing they changed afterward. Weak answers were either "I can't think of one" or a story where the lesson was that other people should have listened to them.

**Reading code before writing it.** We asked people to review a forty-line pull request and say what they'd comment on. The best candidates caught the subtle bug (an off-by-one in pagination) but also flagged the naming and the missing test, and prioritised between them. This one predicted code review quality almost perfectly, which makes sense in retrospect.

## What turned out to be weak signal

**The classic algorithm question.** We had a medium-difficulty problem, the kind involving a hash map and a sliding window. Almost everyone we interviewed had seen something like it. The result was that it measured recent practice, not ability. Two of our strongest hires stumbled on it; two people who aced it were let go from the pipeline at the system design stage for not being able to reason about a queue. We've dropped it.

**"Explain how a database index works."** Everyone gave the same answer, because the same answer is on the first page of every search result. I wanted a question about indexes, so I replaced it with "here's a slow query and the table schema, what would you try?" That version separated people immediately.

**System design at whiteboard scale.** "Design a URL shortener" was near-useless because everyone had rehearsed it. What worked better was starting from a design the candidate already knew, their own current system, and asking what they'd change if traffic went up ten times. That let people show real depth instead of recited boxes.

**Questions about specific frameworks.** I asked about a particular ORM's transaction handling in early interviews. It measured whether people had used that ORM. That's not what we were hiring for, and we ended up passing on a candidate who would have learned it in a week.

## Things that surprised me

- Candidates with the strongest résumés were not, on average, stronger in the loop. The variance within "worked at a big company" was as large as the variance across the whole pool.
- Interview length mattered less than I thought. Our forty-five minute sessions gave nearly the same signal as the hour-long ones, and candidates were noticeably fresher.
- Our panel disagreed with each other most on the questions that turned out to be weak signal. When a question is measuring something real, people tend to agree on what they saw.

## What we changed

We cut the algorithm round, kept debugging and code review, rewrote the design round around the candidate's own experience, and added the "when were you wrong" question to every loop. The process is shorter and I trust it more. I wouldn't claim fifty is a large sample, but it was enough to make me stop defending questions I had never actually validated.
