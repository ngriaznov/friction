# Fifty Backend Interviews in One Quarter: What I Got Wrong

Last quarter I had one headcount to fill: a mid-level backend engineer for our payments platform team. By the time we made an offer I had run fifty technical interviews myself. That's roughly four a week, plus debriefs, on top of my actual job.

I went in with a loop I'd inherited and mostly trusted. I came out having thrown away about a third of it. Here's what I learned.

## Some of my favorite questions were poor signal

**"Design a URL shortener."** Everybody has seen this one. Around the fifteenth interview I noticed the answers were nearly identical: hash the URL, base62-encode it, put a cache in front, mention sharding. Strong and weak candidates gave the same answer because they'd studied the same YouTube video. It told me who had prepared, not who could think.

I replaced it with a design problem taken from our own domain: reconciling a ledger against a payment provider's daily settlement file, where the two sometimes disagree. It has no famous answer. Within about ten minutes I can see whether someone asks what "disagree" means before they start drawing boxes.

**Trivia about language internals.** I used to ask how Go's garbage collector works, or what the GIL actually locks. Candidates who knew the answer weren't noticeably better at the job, and candidates who didn't weren't worse. Two of the strongest people I talked to that quarter guessed wrong on these questions and then reasoned their way to something sensible. I would take that over recall every time.

**The timed LeetCode-style problem.** I kept a version of this in the loop but cut its weight. It correlated well with one thing, which was whether the candidate had recently been grinding practice problems. It correlated weakly with how they did in the debugging exercise, and that was the part of the loop I ended up trusting most.

## What turned out to be strong signal

**Debugging a real (sanitized) bug.** I give candidates a small service with a failing test and a log excerpt. The bug is a race between a retry and an idempotency check, and we actually shipped it to production two years ago. Watching someone read unfamiliar code, form a hypothesis, and test it is the closest I've come to watching them do the job.

**"Tell me about something you broke in production."** This is a behavioral question, but the follow-ups are technical. The good answers had specifics: what the alert said, what they checked first, what they changed afterward. The weak answers were vague or blamed someone else.

**Asking clarifying questions without being prompted.** In every exercise, the candidates who asked "what's the expected volume?" or "is eventual consistency OK here?" did better in the rest of the loop. It was the most reliable single pattern I saw.

## Process lessons

**Write feedback before the debrief.** For the first few weeks I sometimes walked into the debrief with only rough notes. My opinion drifted toward whoever spoke first. Once I made myself write a hire or no-hire, with evidence, within an hour of the interview, my calls got more consistent.

**Calibrate early.** Around interview twenty I realized that two of us on the panel graded the system design round very differently. One person expected capacity math and the other didn't care about it. We should have agreed on a rubric in week one, not week six.

**Fifty is too many for one person.** By the end I was tired and noticed I was less patient with nervous candidates. That isn't fair to them. Next time I'll spread the load across more interviewers even if it makes calibration harder.

## Where we landed

We hired someone who, frankly, did poorly on the timed coding problem. She took the lead on the debugging exercise, asked six clarifying questions in the design round, and had a very good story about taking down a staging database. She's been on the team for two months and has already fixed a reconciliation bug that had been open for a year.

The loop we use now is shorter, relies more on our own problems, and is less comfortable for candidates who have only studied standard prep material. I think it's better. I'll know more after the next fifty.
