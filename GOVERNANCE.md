# TorchDR governance

TorchDR is a community-maintained open source project. This document defines
how the project is governed, how decisions are made, and how maintenance
responsibility is transferred.

## Roles

### Contributors

Anyone who reports issues, proposes changes, improves documentation, reviews
pull requests, or otherwise helps the project is a contributor. Contributors
do not need a formal invitation or recurring commitment.

### Maintainers

Maintainers are responsible for the health and continuity of the project. They:

- review and merge changes;
- triage issues and releases;
- maintain compatibility, testing, and documentation;
- uphold the Code of Conduct; and
- steward the project in the interests of its users and contributors.

The current maintainers are:

| Maintainer | GitHub handle |
| --- | --- |
| Hugues Van Assel | [@huguesva](https://github.com/huguesva) |
| Rémi Flamary | [@rflamary](https://github.com/rflamary) |

Repository ownership is encoded in [`.github/CODEOWNERS`](.github/CODEOWNERS).

## Decision making

TorchDR uses consensus-seeking decision making in public GitHub issues and pull
requests.

- Routine bug fixes, documentation changes, and maintenance work may be merged
  after approval from one maintainer other than the author, when practical.
- Changes to public APIs, compatibility guarantees, governance, licensing, or
  project direction should first be discussed in an issue. Maintainers will
  allow enough time for affected contributors to comment before deciding.
- Releases require approval from a maintainer and must satisfy the published
  release criteria.

When consensus cannot be reached, the maintainers may call a vote. Each active
maintainer has one vote, and a strict majority decides. A tied vote preserves
the status quo while the maintainers seek more community input. For an urgent
security, compatibility, or operational problem, maintainers may take the
smallest reversible action needed and document it afterward.

## Conflicts of interest and conflict resolution

Maintainers and contributors must disclose material conflicts of interest and
withdraw from decisions when their impartiality could reasonably be questioned.

Technical disagreements should be resolved by documenting the alternatives,
tradeoffs, and available evidence in the relevant issue or pull request. If the
maintainers remain deadlocked, they should jointly invite an experienced,
neutral contributor or subject-matter expert to advise. The status quo remains
in effect if no resolution is reached.

Conduct concerns are handled under the
[Code of Conduct](CODE_OF_CONDUCT.md), including its private reporting path.

## Appointing maintainers

A contributor may be nominated by an existing maintainer after demonstrating
sustained, constructive participation and sound judgment. The nomination must
be discussed by the active maintainers, accepted publicly by the candidate,
and approved unanimously by the other active maintainers. Appointment grants
the repository permissions required for the role and adds the maintainer to
this document and `CODEOWNERS`.

## Stepping down, removal, and emeritus status

A maintainer may step down at any time. They will be moved to emeritus status
unless they ask not to be listed. Emeritus maintainers retain recognition for
their service but have no required review, release, or decision-making duties.
They may return through the normal appointment process.

A maintainer may be removed for a serious Code of Conduct violation, sustained
failure to protect the project, or prolonged inactivity. Removal requires the
unanimous agreement of all other non-conflicted active maintainers. Before an
inactivity removal, the project will make a reasonable private contact attempt
and allow at least 30 days for a response. Access may be suspended immediately
when necessary to protect users or the repository.

## Succession and continuity

The project aims to retain at least two active maintainers. When only one
remains, recruiting and onboarding another maintainer becomes a priority. An
outgoing maintainer should, when possible, announce the transition, transfer
release and administrative knowledge, and help identify a successor.

If no maintainer remains active, the TorchDR GitHub organization owners may
appoint one or more interim maintainers after opening a public issue and
allowing at least 14 days for community comment. Interim maintainers use the
normal appointment process to establish a durable maintenance team.

## Amendments

Governance changes follow the decision process above and must be proposed by
pull request. Material changes should be announced in a linked public issue.
