# Generator distribution check: synthetic generator vs PicoDomain vs published values

Date: 2026-09-21. Branch `fix/generator`. Generator: `src/data/stream_synthetic.py` with the published `TGNConfig`
(50 registered users, 1000 guests, 80 devices, 150 sources, 40 configs, 1000 resources, 200k events).
Dev seeds 2000/2001/2002. No file under src/, tests/ or docs/ was modified.

Scripts:
- `tasks/tmp/dist_check.py`: all statistics for the generator and PicoDomain. Output is in `tasks/tmp/dist_check_{gen2000,gen2001,gen2002,pico}.json`.
- `tasks/tmp/dist_check2.py`: within-regime burstiness and a `p_hotdesk=0` ablation, seed 2000.
- PicoDomain goes through `tests/datasets/picodomain.load_picodomain_stream` (bind_ttl 36000 s, ±90 s label window): 55,436 access events, 2.67 days. Raw Zeek `conn`/`ssl`/`kerberos` were also parsed on the host.

Conventions:
- Ranges are seed 2000–2002.
- "Human users" means registered users 1..49. The user-0 service account and the guests are excluded unless stated otherwise.
- Unless a row says otherwise, fan-out, Zipf and activity statistics use benign events only (etype 0).

## 1. Resource popularity

| Statistic | Generator | PicoDomain | Literature | Verdict |
|---|---|---|---|---|
| Global Zipf s, discrete MLE on rank-frequency | 1.12–1.21 | 1.45 | 0.64–0.83 for web proxy traces [1] | Outside the web-proxy range, below PicoDomain. The generator realises its configured s=1.2 |
| Global Zipf, OLS log-log, top-100 ranks | 1.15–1.18 | 1.30 | 0.64–0.83 [1] | Same |
| Top-10 resources' share | 49–58 % | 78 % | n/a | Plausible |
| Per-user Zipf s, MLE, median (p10–p90) | 1.16–1.21 (1.02–1.46) | 0.98 (0.76–1.68, n=10) | not found | Plausible, but per-user dispersion is narrow |
| Distinct resources per user, median | 238–277 of 1000 | 30 | not found | Not comparable (catalogue size is a free parameter) |

Notes:
- [1] measures web-proxy object popularity. An internal ZTA resource catalogue is a different population, so s=1.2 is defensible only as a declared choice. The docs already say so (§2.2).
- The per-user exponent is the same law for everyone. Only the habitual subset (half of the role's valid actions) differs between users.

## 2. Inter-arrival times

| Statistic | Generator | PicoDomain | Literature | Verdict |
|---|---|---|---|---|
| Global gap mean / median | 132 s / 37 s | 4.2 s / 1 s | n/a | Not comparable (different event granularity) |
| Global CV / burstiness B=(σ−μ)/(σ+μ) | 2.83–2.89 / 0.48 | 5.20 / 0.68 | Human activity has high B, driven by the inter-event distribution, not memory [3] | Superficially plausible, but see the next row |
| CV / B inside one regime (weekday 09–17) | 1.01 / 0.006 | 5.0 / 0.67 | Poisson holds for user **session** arrivals at fixed hourly rates. Exponential gaps "grievously underestimate the burstiness" of within-session arrivals [2] | **Outside range.** Global burstiness comes only from the 3 regime switches; each regime is exactly Poisson |
| Per-user CV / B, whole stream, median | 4.2 / 0.61 | 11.0 / 0.83 | [3] | Plausible only because of regime mixing |
| Per-user CV / B, weekday 09–17, median | 1.08 / 0.04 | n/a | [3]: human activity has B well above 0 | **Outside range** |
| Hourly profile | Step function: 08–18 ≈9 %/h, other hours ≈0.8 %/h (12× peak/trough). No ramps, no lunch dip | Nearly flat to 15:00 (automated lab), then drops (26×). Raw conn.log is flat 4–6 %/h | [2] Fig. 1 shows smooth diurnal hourly rates (qualitative) | Plausible to first order, but visibly synthetic |
| Weekday vs weekend daily volume | 881 vs 71 events (12:1). Day-to-day CV of weekday volume 0.04 | 1 weekday + 2 weekend days, not derivable | not found | Weekday CV 0.04 is implausibly constant (every weekday is the same Poisson draw) |

## 3. Per-user activity heterogeneity (human users, all event classes)

| Statistic | Generator | PicoDomain | Literature | Verdict |
|---|---|---|---|---|
| Events per user, p10 / median / p90 | ~1.2k / 2.6–2.7k / 5.4–6.4k | 2 / 1.1k / 4.0k | Human activity is heavy-tailed [3] (qualitative). No verified per-user enterprise number found | |
| Gini / CV / max÷median | 0.33–0.35 / 0.62–0.70 / 3.2–4.4 | 0.78 / 2.4 / 34 | not found | **Too homogeneous** |
| Weekday daily active users | 43 of 49 | n/a | not found | Plausible |

Cause: `step()` picks a machine uniformly. A user's volume is then set only by how many machines they own or share, with no per-user activity rate.

## 4. Fan-out (benign, human users)

| Statistic | Generator | Generator, `p_hotdesk=0` (seed 2000) | PicoDomain | Literature | Verdict |
|---|---|---|---|---|---|
| Distinct devices per user, median (p90) | **50–52** (55–62) | 7 | 1 (2) | not found (BYOD/shared-device rates: UNVERIFIED, no source found) | **Outside range** |
| Share of a user's events on their top device, median | 0.54 | 0.49 | n/a | n/a | Too low |
| Users per device, median (max) | **9–11** (47–48) | 1 | 1 (4) | not found | **Outside range** |
| Devices with more than 1 user | **94–96 %** | n/a | 42 % (includes the red-team account that is on every host) | not found | **Outside range** |
| Distinct IPs per user, median | **192–206** of 150 home + 6000 fresh | 191 | 1 | not found | **Outside range** |
| IPs per device, median | 40–47 | n/a | 1 | not found | Outside range |
| Distinct JA3 per user, median | **36–38** (of 40 habitual) | 17 | 2 | not found | **Outside range** |
| Distinct JA3 per device, median (max) | 3 (8–9) | n/a | 2 (5). Raw ssl.log: 1–5 per host | 1.5–3.2 fingerprints per *process* (browser 3.18) [6] | Plausible |
| Fleet JA3: distinct benign / top-1 share / configs covering 90 % | 75–90 / 18 % / 40–43 | 14 / 76 % / 3 | 7,909 endpoint fingerprints for about 24k users. 22–77 processes per fingerprint, i.e. heavy sharing [6] | **Too flat.** Real fleets concentrate on a few stacks (PicoDomain). Uniform `np.random.choice(cfg_pool)` gives about equal shares |

Causes, from reading `stream_synthetic.py`:
- `p_hotdesk=0.02` puts the user on a machine drawn **uniformly from all 80 machines** (lines around 1028–1030). Over about 4,000 hot-desk events in 306 days, every user touches about 50 machines and 36 of the 40 JA3s. The ablation shows hot-desking alone accounts for 52→7 devices per user and 9→1 users per device.
- Roaming draws a **uniform** IP from the whole 150-IP pool (`np.random.randint(0, num_sources)`). Users therefore appear from each other's home or DSL IPs, which gives about 200 IPs per user.
- The remaining 7 devices per user at `p_hotdesk=0` come from round-robin ownership of 80 machines by 49 users, 20 % shared machines with 1–3 extra users, and cookie wipes (`p_cookie_wipe=0.001`, about 200 re-keys).

## 5. Novelty: share of events whose node was never seen before

The table uses seed 2000. Seeds 2001 and 2002 are within ±0.01 on every cell (see the JSON files).

| Split / class | Source | Config | Device | User | Any of the 4 |
|---|---|---|---|---|---|
| Train, benign | 2.95 % | 0.03 % | 0.10 % | 0.77 % (guests) | 3.80 % |
| Val, benign | 2.68 % | 0.03 % | 0.04 % | **0** | 2.74 % |
| Test, benign | 2.76 % | 0.03 % | 0.04 % | **0** | 2.82 % |
| Train, theft | 13.1 % | 5.8 % | 13.6 % | 0.7 % | 20.3 % |
| Test, theft | 9.7–12.8 % | 5.9–7.2 % | **0** | 0 | 12–17 % |
| Test, lateral | 1.2–3.1 % | **0** | 0 | 0 | 1.2–3.1 % |
| Test, contextual / exfil / policy | 0–3 % | 0 | 0 | 0 | 0–3 % |
| PicoDomain, all splits, benign | 0.01–0.04 % | 0.02–0.07 % | ≤0.04 % | ≤0.04 % | ≤0.07 % |
| PicoDomain, test, theft / lateral | 0 | 0 | 0 | 0 / 0.3 % | 0–0.3 % |

Other novelty statistics:
- New benign entities per day, second half of the stream: 17.5 IPs, 0.12–0.19 JA3, 0.34–0.41 devices, 0 users.

Literature for comparison:
- New-IP rate: no verified enterprise figure found (UNVERIFIED).
- Onboarding: the US hires rate is 3.2 %/month and the separations rate 3.2 %/month (BLS JOLTS, July 2026) [8]. Over the ~92-day test window that means roughly 10 % of the workforce should be new (~5 of 50 users). The generator has **0 new registered users after day 0, and no departures**.

Verdicts:
- New-IP share, 2.8 % of benign events: plausible but UNVERIFIED.
- User churn: **outside range**, because the population is closed.
- Device novelty: **inconsistent across splits.** The fresh-device pool (`num_wipe_slots + num_theft_slots` = 192) is recycled round-robin. Device novelty therefore falls to 0 in val and test, and theft goes from 13–17 % new devices in train to 0 % in test. This is a train/test shift created by the generator, not by the attacker model.
- Config novelty: **a residual shortcut.** In test, a never-seen JA3 is about 200× more frequent in theft (6–7 %) than in benign traffic (0.03 %). A never-seen IP is about 4× more frequent in theft.
- Guest users: all 1000 guests first appear in train (they are about 14 % of events and recur all year). Real anonymous visitors keep arriving, so user novelty should not be 0 in test.

## 6. Attack prevalence and kill chain

| Statistic | Generator | PicoDomain | Literature | Verdict |
|---|---|---|---|---|
| Attack prevalence, etype 1–5 | 1.33–1.40 % | 1.57 % | LANL: 749 red-team rows / 1,051,430,459 auth rows = 7.1e-7 [4a]. Euler: 518 anomalous edges / 45,871,390 events = 1.1e-5 [5]. CIC-IDS2017: 19.7 % [9] | Plausible for a **benchmark** (between LANL and CIC-IDS, close to PicoDomain). Not an operational base rate, 3–4 orders of magnitude above LANL |
| label=1 share including benign-denied | 2.9–3.0 % | n/a | n/a | n/a |
| Intrusions per day, 80-machine fleet | 0.35–0.41 (107–126 per 306 days, i.e. each machine compromised ~1.4×) | 1 campaign / 2.67 days | LANL: one red-team campaign over 58 days on 17,684 computers [4] | **Outside range** (intrusion frequency per host) |
| Compromise to remediation duration, median (p90, max) | 65–71 h (130–147 h, ≤10 d) | ~2 days (red log) | Global median dwell time 11 days in 2024, 10 days in 2023 [7] | **Short**, by about 4× |
| Attack events per intrusion (recon 1–3, lateral 5–11, exfil 1–3, plus dwell) | 14.5–14.7 | 870 labelled events for 1 campaign | not found | Not comparable |
| Credential-theft incidents per day / events per incident | 0.67–0.74 / 3–6 (median 5) | 102 theft-labelled events | not found | UNVERIFIED |

## Deviations a reviewer is most likely to attack, in priority order, with the fixing knob

1. **Identity fan-out is unrealistic** (≈50 devices, ≈36 JA3 and ≈200 IPs per user; 95 % of devices shared; PicoDomain median is 1/2/1).
   - Fix `p_hotdesk`: draw the hot-desk machine from a small fixed per-user set (2–3 neighbour or meeting-room machines) instead of `random.choice(self._humans)` on a uniform machine, or lower it to ≤0.005.
   - Fix roaming: give each machine a small personal roaming set (home, mobile) plus the fresh pool (`p_new_source`), instead of a uniform draw over `num_sources`.
2. **Poisson requests with no burstiness** (within-regime CV 1.01, B≈0; PicoDomain B 0.67; [2][3]).
   - Replace `_current_interarrival_scale` plus exponential gaps with a session layer: Poisson session starts at an hourly rate, heavy-tailed (lognormal or Pareto) gaps inside a session, and heavy-tailed session length.
   - Re-run the Δt leakage audit afterwards, because log1p(Δt) is an edge feature.
3. **Homogeneous user activity** (Gini 0.34 vs 0.78 for PicoDomain).
   - Add a per-user lognormal activity rate: pick the user, then the machine, instead of a uniform machine draw.
4. **Closed population, and slot recycling kills device novelty in test.**
   - Add `num_new_users` with a hires/separations rate of about 3 %/month [8].
   - Size `num_wipe_slots + num_theft_slots` (or make the allocator grow) so fresh devices are never recycled.
   - Give guests fresh ids per session.
5. **Config novelty is still a class tell in test** (6–7 % for theft vs 0.03 % for benign).
   - Raise `p_config_release` and/or `p_config_adopt`, or raise `p_theft_mimic_config` (0.7 now), so that benign never-seen JA3 is comparable to the non-mimic theft share.
6. **Fleet JA3 distribution is too flat** (40 configs cover 90 %; PicoDomain: 3).
   - Draw `machine_configs` from a Zipf over `cfg_pool` instead of uniformly.
7. **Kill-chain timing**: about 1.4 compromises per machine in 10 months, and dwell of about 2.8 days vs 11 days [7].
   - Lower `p_compromise`. Stretch recon and lateral duration (dwell counted in time, not events).
   - State plainly in the paper that 1.3 % is a benchmark prevalence, not an operational one [4a][5].
8. **Zipf s=1.2 is above the web-proxy range** [1]. This is already disclosed in docs §2.2.
   - Either keep it (PicoDomain 1.45 supports a higher s for internal resources) or ablate `s ∈ {0.8, 1.2}`. The exponent is hard-coded in `ZTAStreamSimulator.__init__` (`** 1.2`) and should become a `TGNConfig` field.
9. **Diurnal shape is a step function** with constant weekday volume (CV 0.04).
   - Use a smooth hourly profile and a per-day lognormal volume multiplier.

## References (verified unless marked)

- [1] L. Breslau, P. Cao, L. Fan, G. Phillips, S. Shenker, "Web Caching and Zipf-like Distributions: Evidence and Implications", IEEE INFOCOM 1999, pp. 126–134. Verified in the author's PostScript https://pages.cs.wisc.edu/~cao/papers/zipf-like.ps.gz: "value of α varies from trace to trace, ranging from 0.64 to 0.83".
- [2] V. Paxson, S. Floyd, "Wide Area Traffic: The Failure of Poisson Modeling", IEEE/ACM Trans. Networking 3(3), 1995, DOI 10.1109/90.392383. Verified text: https://web.stanford.edu/class/cs244/papers/paxson1995.pdf ("connection arrivals are well-modeled as Poisson with fixed hourly rates … exponentially distributed interarrivals … grievously underestimate the burstiness").
- [3] K.-I. Goh, A.-L. Barabási, "Burstiness and Memory in Complex Systems". Verified at https://arxiv.org/abs/physics/0610233: B is defined there, and human activity (email, library, printing, phone) has high Δ=B with negligible memory. Venue EPL 81:48002 (2008) is **UNVERIFIED**. No numeric B was taken from the figure.
- [4] A. D. Kent, "Cybersecurity Data Sources for Dynamic Network Research", in *Dynamic Networks in Cybersecurity*, Imperial College Press, 2015. Dataset page https://csr.lanl.gov/data/cyber1/ (verified: 58 days, 1,648,275,307 events, 12,425 users, 17,684 computers).
- [4a] Per-file row counts (auth 1,051,430,459; redteam 749, of which 12 are duplicates): G-Research, https://github.com/G-Research/dgraph-lanl-csr README. This is a secondary source, verified there.
- [5] I. J. King, H. H. Huang, "Euler: Detecting Network Lateral Movement via Scalable Temporal Link Prediction", NDSS 2022. Verified in https://www2.seas.gwu.edu/~howie/publications/Euler-NDSS22.pdf, Table V: 17,685 nodes, 45,871,390 events, 518 anomalous edges, 58 days.
- [6] B. Anderson, D. McGrew, "TLS Beyond the Browser: Combining End Host and Network Data to Understand Application Behavior", ACM IMC 2019, DOI 10.1145/3355369.3355601. Numbers verified in the authors' IETF 106 MAPRG slides, https://datatracker.ietf.org/meeting/106/materials/slides-106-maprg-tls-beyond-the-browser-00: ~24,000 users; 7,909 endpoint fingerprints; fingerprints per process 1.53–3.18; processes per fingerprint 22–77.
- [7] Mandiant, "M-Trends 2025", Google Cloud blog, https://cloud.google.com/blog/topics/threat-intelligence/m-trends-2025 (verified: "Global median dwell time rose to 11 days from 10 days in 2023").
- [8] U.S. BLS, Job Openings and Labor Turnover Survey, July 2026, https://www.bls.gov/news.release/jolts.nr0.htm (verified: hires rate 3.2 %, total separations rate 3.2 %).
- [9] A. Ahmim et al., "A Novel Hierarchical Intrusion Detection System based on Decision Tree and Rules-based Models", arXiv:1812.09059 (venue DCOSS 2019 **UNVERIFIED**). Verified there: CICIDS2017 has 2,830,743 rows, 2,273,097 of them BENIGN, i.e. 19.7 % attack.
- Not found or UNVERIFIED: an enterprise new-IP or roaming login rate; BYOD or shared-device prevalence; a numeric per-user activity Gini for enterprise auth logs; credential-theft session length.
