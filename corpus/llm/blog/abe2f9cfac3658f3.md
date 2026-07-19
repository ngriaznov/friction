## From Red Brick to Cloud: Migrating Our Rails App's Background Jobs from Sidekiq + Redis to AWS SQS

For a long time, our Ruby on Rails application, “Project Phoenix,” relied heavily on background jobs for tasks like sending weekly newsletters, processing large image uploads, and generating daily reports. We’d built this system using Sidekiq, paired with a Redis backend for queuing the jobs. It was working… mostly. As we scaled, however, cracks started to show – primarily around reliability, observability, and cost. This blog post details our decision to migrate from this familiar setup to Amazon SQS, the challenges we faced during the transition, and how things unfolded in the crucial first week after going live.

**Why Leave the Brick Behind? The Rationale for Switching**

Our initial concerns with the Sidekiq + Redis architecture were mounting. Firstly, managing our own Redis cluster was becoming increasingly complex – patching, scaling, ensuring high availability… it all added overhead that wasn't directly contributing to Project Phoenix’s core value proposition. Secondly, while Sidekiq offered decent throughput, we were experiencing occasional slowdowns during peak processing times, particularly with the image uploads. Finally, and perhaps most significantly, the cost of running a dedicated Redis cluster was steadily rising.

We knew there had to be a better way – one that leveraged cloud-native services for scalability and reduced operational burden. AWS SQS (Simple Queue Service) immediately jumped out as the obvious solution. It offered built-in durability, automatic scaling, and a pay-as-you-go model that aligned perfectly with our needs.  The core concept was simple: offload job queuing to a managed service designed for exactly this purpose.

**The Migration – A Carefully Planned (But Not Perfectly Executed) Process**

We adopted a phased migration strategy, prioritizing jobs based on their volume and impact. We started with the newsletter sending jobs - relatively low-impact and predictable.  This served as a crucial proof of concept and allowed us to iron out many initial issues. 

Here’s a breakdown of our key steps:

* **Infrastructure Setup:** We provisioned an SQS queue for each job type, leveraging AWS IAM roles for secure access from our Rails application.
* **Code Modifications:** We refactored the code that triggered jobs to interact with SQS using the `sidekiq-sqs` gem. This gem handles the complexities of sending and receiving messages between Sidekiq and SQS.
* **Testing (Extensive):**  We ran a suite of integration tests, mimicking production load to identify potential issues before deploying anything live.
* **Staged Rollout:** We gradually increased the number of jobs routed through SQS while keeping the old Redis-backed system running in parallel. This allowed us to monitor performance and quickly revert if necessary.


**What Broke? The Unexpected Hiccups**

Despite careful planning, things weren’t entirely smooth sailing.  The biggest issue we encountered was around message acknowledgement. Sidekiq's approach to acknowledging job completion (using a callback) didn’t translate directly to SQS. With SQS, messages are only considered “consumed” after an explicit acknowledgment is sent back to the queue. This introduced potential race conditions where a job might be processed multiple times if acknowledgment wasn’t handled correctly. 

We resolved this with careful implementation of the `sidekiq-sqs` gem's acknowledgement handling and adding retry logic in case of transient errors.  Another minor issue involved occasional delays in message delivery, which we traced back to network latency between our application servers and AWS. We adjusted our server locations to minimize these distances. Finally, we discovered that some jobs, particularly the image upload processing, were significantly slower on SQS due to the nature of SQS – it’s designed for asynchronous processing, not necessarily high-speed data transfer.

**The First Week: On-Call Load & Lessons Learned**

The first week after the cutover was a baptism by fire. Our on-call team (consisting of two developers) experienced a significantly higher workload than anticipated. 

Here's a breakdown:

* **Increased Alerting:**  The SQS queues generated numerous alerts related to message delivery failures and processing delays. We quickly learned that SQS’s monitoring tools weren’t as intuitive as Sidekiq’s, requiring us to build custom dashboards for real-time visibility.
* **Manual Intervention:** Several times, we had to manually intervene to investigate and resolve issues – restarting failed jobs, adjusting queue settings, and even temporarily reverting to the Redis backend during particularly stressful periods.
* **Longer Response Times:**  While overall throughput was comparable, response times for certain operations (specifically image processing) were noticeably longer due to the increased latency involved in moving data between our servers and SQS.

**Key Takeaways & Moving Forward**

The migration to AWS SQS wasn’t a simple swap; it forced us to rethink how we approached background job management. We learned valuable lessons about message acknowledgement, monitoring, and the inherent differences between managed queue services and self-hosted solutions. 

Moving forward, we are:

* **Investing in Enhanced Monitoring:** Building more sophisticated dashboards to proactively identify potential issues.
* **Optimizing Image Processing:** Exploring strategies for faster image processing, potentially leveraging AWS Lambda or other serverless technologies.
* **Refining our Rollout Strategy:**  We’re planning a more gradual rollout across all job types, incorporating lessons learned from the newsletter migration.

Ultimately, the switch to SQS was a crucial step in modernizing Project Phoenix and setting us up for sustainable growth – albeit one that highlighted the importance of meticulous planning and a healthy dose of preparedness when venturing into the cloud. 


---

Would you like me to elaborate on any specific aspect of this blog post, such as:

*   The code changes involved?
*   A more detailed breakdown of the monitoring tools we built?
*   Specific strategies for optimizing image processing?
