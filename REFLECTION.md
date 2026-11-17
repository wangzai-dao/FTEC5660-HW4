# Reflection — FTEC5660 Homework 4

Individual homework. Task 6 of the assignment: what I want to be in a world where personal AI
agents can do most of the work.

> *Imagine that ten years from now, everyone has access to powerful personal AI agents capable of
> writing, programming, conducting research, managing businesses, and making sophisticated decisions.*
>
> The seven questions of Task 6, answered in order.

## What kind of person would I want to be in that world?

I would want to be the person who still knows what the question is. In this homework the model that
writes the answer was given to us, and it was never the bottleneck: what decided whether the answer
was right were a hundred small judgements around it — which row belongs to which column, whether the
FY2025 report is the report of FY2025, whether "we cannot answer this" is the honest answer. Ten years
from now I would like to be good at those judgements: not the one who produces the most, but the one
who can say what is worth producing, what a correct answer looks like, and what would make it wrong.
And I would like to remain a person with a life outside the work — someone who cooks, runs, sits with
friends without a screen. In a world where my ability to produce is unremarkable, the things that
make me *me* are more likely to be found in attention than in output.

## Which of my current ambitions would still matter?

The ambition to *understand*. Not to have read the paper, but to be able to explain why the method
works and when it fails; the difference between those two shows up the moment something goes wrong,
and something always goes wrong. The ambition to be trusted: I would like the people I work with to
assume that if I say a number is right it has been checked, and that if I do not know, I say so. And
the ambition to build something that outlives the demo — most of what I have built so far is a demo;
the harder and more durable thing is a system that someone else can operate, audit and repair after I
leave. What would matter less: being the fastest, being in the room, being impressive in the first
five minutes. An agent can do the first five minutes.

## What would I want to be known for beyond productivity and professional achievements?

For judgement in the presence of ambiguity, and for being honest about uncertainty. This assignment
gave a strange, formal name to that virtue: abstention. The system earns ten points for answering
`NOT_FOUND` when the library cannot answer, because in the private set it loses more by guessing than
by refusing. People are graded the same way, only less explicitly. I would like to be the colleague
who says "we do not have that number" instead of the one who supplies a plausible one. Beyond that,
for generosity: the parts of my education that changed me were taught by people who explained things
they did not have to explain, and I would like to be that person for someone. And for taste — to
prefer the plain correct sentence to the impressive vague one.

## What am I willing to struggle for, even when an agent could make the struggle unnecessary?

Learning things slowly. I could ask an agent to summarise every paper I read; I also know that the
summary is not the thing I wanted. Reading a proof, deriving something badly and then correctly,
writing a paragraph five times — that is how I build the model of the world that lets me notice when
the summary is wrong. The same goes for the craft of verification: the habit of asking "how would I
know this is false?" before using an answer. And relationships, which cannot be delegated at all:
being present for my family, for my friends, for the people I work with. And physical practice —
running, cooking, sleeping properly. Those are the parts of a life an agent cannot hold for me.

## Which skills, qualities, or experiences would still make me valuable?

Framing: turning a vague business worry into a question that can be answered and, better, measured.
Evaluation: knowing what "good" means, building the ground truth, and measuring honestly — the
ablation table in this report is the most transferable artefact I built this semester, because it is
machinery for knowing whether a change helped. Domain literacy in accounting and finance: a model will
happily give me a number, but knowing that "net accounts receivable" is not "revenue", that an FY2025
report also contains FY2024, that a target price is not a forecast of the share price, is what lets me
catch the error. And responsibility: someone has to sign.

## If my value is no longer determined primarily by what I can do, how will I define who I am?

By what I notice, what I care about, whom I show up for, and what I refuse to do. Being a person is
not a job description. I would define myself by the questions I choose to spend my life on, by the
standard I hold when nobody is checking, and by the people who would say I was there when it mattered.
That sounds sentimental, but I think it is the only definition that survives a world in which an agent
can do my work: the doing can be substituted, the caring cannot.

## What should I start doing differently tomorrow?

Three things. Keep a verification habit: for anything I claim, write down how I checked it and what
would falsify it — the way this homework records its evidence pairs. Spend more time on the problem
before the solution: I lost far more time to badly framed retrieval than to badly written code, and
the same is true of most projects. And keep one practice no agent can do for me, plus one relationship
I invest in with no productivity reason. Concretely, before the next project I will write the ground
truth and the evaluation *first*, even a small one, instead of at the end — this time the numbers I
trusted most were the ones I could recompute from the evidence file, and that habit generalises well
beyond this course.
