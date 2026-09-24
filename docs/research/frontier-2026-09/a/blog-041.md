# What I Wish I'd Known Before My First Conference Talk

Last spring I gave my first talk at a technical conference: thirty minutes on how our team migrated a large Postgres database with almost no downtime. It went fine. Not great, but fine. Most of the gap between fine and great came from things nobody told me in advance, so here they are.

## Slides

**Write the talk before you make the slides.** My first draft was 58 slides built in the order I thought of things. It had no argument, just a timeline. Eventually I threw it out, wrote a one-page outline with a single sentence for the main takeaway, and only then opened the slide tool. The second deck was better in every way.

**One idea per slide, and fewer slides than you think.** A rough rule that worked for me is about one slide per minute. I ended at 31 for a 30-minute slot, and even that was a bit rushed.

**Code on slides should be huge and short.** I had a 25-line SQL snippet at 14-point font. On the venue projector nobody past the fourth row could read it. Aim for eight lines or fewer at 28 point or larger, and highlight the one line that matters.

**Assume the projector will betray you.** Test your slides at 1024×768 and in low contrast. Bring your deck on a USB stick and as a PDF. Bring your own adapter. My dark-theme syntax highlighting disappeared on the venue's washed-out projector, and I spent the first five minutes apologizing for it.

## Rehearsal

**Rehearse out loud, standing up, with a timer.** Reading through your slides in your head doesn't count. My first real run-through took 44 minutes, not 30. Silent run-throughs hide how long it takes to actually say the words.

**Do at least one run in front of people.** I gave the talk to four coworkers over lunch. They found two sections that only made sense if you already knew our system, and a joke that didn't land. I cut both.

**Memorize the first two minutes.** Nerves are worst at the start. With a scripted opening, I was on autopilot until my heart rate settled. After that, the slides carried me.

**Plan what to cut if you're running long.** I marked two slides as skippable in advance. When I saw I was three minutes behind, I skipped them without panicking.

## Questions afterward

**Repeat every question into the microphone.** The audience usually can't hear the person asking, and the recording definitely can't. It also gives you a few seconds to think.

**"I don't know" is a complete answer.** Someone asked about a replication edge case I'd never tested. I said I didn't know, that I'd guess X, and that I'd be happy to dig into it with them afterward. We talked for twenty minutes in the hallway, and it was the best conversation of the conference.

**Some questions are really comments.** Now and then someone uses the Q&A to give a mini talk of their own. Thank them, say it's an interesting point, and take the next question. You don't need to argue.

**Leave time.** I finished right at 30 minutes and had to take questions in the hallway. Next time I'll aim for 25 minutes of content.

## The thing I most wish I'd known

The audience wants you to do well. Nobody in that room was waiting for me to fail. Once I actually believed that, about ten minutes in, the rest of the talk was almost fun.
