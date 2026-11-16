"""Self-authored fictional retrieval cases; fixed before model outputs are inspected."""
import hashlib
import json

PAIRS = [
 ('reset-password', 'To reset a forgotten password, request a reset link at the sign-in page.', 'I forgot my login secret. How can I get back in?'),
 ('two-factor', 'Two-factor authentication requires a one-time code from your authenticator app.', 'Why must I enter a code after typing my password?'),
 ('delete-account', 'Account deletion permanently removes the user profile after confirmation.', 'I want to erase my profile permanently.'),
 ('change-email', 'You can update the email address associated with your account in profile settings.', 'Where do I replace the address used to receive account messages?'),
 ('refund', 'Refunds return the purchase amount to the original payment method after approval.', 'Can the money for my purchase be sent back to my card?'),
 ('duplicate-charge', 'Duplicate charges for the same purchase should be reported to billing support.', 'My card was billed twice for one order.'),
 ('invoice', 'Download a PDF invoice from the billing history page.', 'Where can I get a receipt file for my payment?'),
 ('cancel-subscription', 'Cancel your recurring subscription before the next renewal in the billing settings.', 'How do I stop automatic monthly renewals?'),
 ('tracking', 'Shipment tracking shows the current location and delivery progress of a parcel.', 'Where is the package I ordered right now?'),
 ('change-address', 'A shipping address can be changed before the order is dispatched.', 'Can I send my purchase to a different house before it ships?'),
 ('damaged-item', 'Report damaged items with photos of the product and packaging to request a replacement.', 'The box arrived broken and the product is cracked. What evidence should I send?'),
 ('return-window', 'Products may be returned within thirty days of delivery under the return policy.', 'How many days do I have to send an unwanted item back?'),
 ('api-key', 'Create an API key in the developer console and store it outside source control.', 'Where do I create a credential for programmatic access?'),
 ('rate-limit', 'A 429 response means the API request rate limit was exceeded; retry with backoff.', 'The service says too many requests. Should I wait before retrying?'),
 ('timeout', 'An upstream timeout occurs when a dependent service does not respond before its deadline.', 'A dependency took too long to reply and my request expired.'),
 ('webhook', 'Webhook events notify your endpoint when an order status changes.', 'How can my server receive a notification after an order update?'),
 ('csv-export', 'Export filtered table rows as a CSV file for spreadsheet analysis.', 'I need the current data table in a format Excel can open.'),
 ('backup', 'Scheduled backups preserve copies of the database for later disaster recovery.', 'How can we restore our records if the database is lost?'),
 ('permissions', 'Role-based permissions restrict which users can view or edit workspace resources.', 'Can I let a teammate read files without allowing changes?'),
 ('audit-log', 'Audit logs record who changed a resource and when the action happened.', 'I need to find who edited this record yesterday.'),
 ('disk-space', 'Free disk space by removing unused temporary files and old local caches.', 'My computer storage is almost full. What can I clean?'),
 ('cpu-load', 'High CPU utilization can slow requests when too many inference workers compete for cores.', 'Why does adding many model workers make my processor busy and responses slow?'),
 ('batching', 'Dynamic batching combines nearby inference requests into one model call.', 'Can several incoming prediction jobs be processed together?'),
 ('quantization', 'INT8 quantization reduces the precision of supported model operations and may change outputs.', 'Will compressing model weights to eight bits affect predictions?'),
]


def corpus():
    return [{'id': i, 'text': d} for i, d, _ in PAIRS]


def queries():
    return [{'id': 'q-' + i, 'text': q, 'relevant_id': i} for i, _, q in PAIRS]


def workload_hash():
    return hashlib.sha256(json.dumps({'pairs': PAIRS, 'performance_texts': performance_texts()}, ensure_ascii=False).encode()).hexdigest()


def performance_texts():
    # A fixed mixed-length workload; no content is sent to any external inference service.
    short = [d for _, d, _ in PAIRS[:8]]
    long = [d + ' ' + ' '.join(x[1] for x in PAIRS[16:19]) for _, d, _ in PAIRS[8:16]]
    return short + long


def ranking_metrics(scores):
    import numpy as np
    scores = np.asarray(scores)
    if scores.shape != (len(PAIRS), len(PAIRS)) or not np.isfinite(scores).all():
        raise ValueError('Need a finite query-by-document score matrix')
    ranks = []
    failures = []
    docs, qs = corpus(), queries()
    for index, row in enumerate(scores):
        order = np.argsort(-row, kind='stable')
        rank = int(np.where(order == index)[0][0]) + 1
        ranks.append(rank)
        if rank != 1:
            failures.append({'query': qs[index], 'top_document': docs[int(order[0])], 'relevant_rank': rank})
    return {'queries': len(qs), 'recall_at_1': sum(r == 1 for r in ranks) / len(ranks),
        'recall_at_3': sum(r <= 3 for r in ranks) / len(ranks), 'mrr': sum(1 / r for r in ranks) / len(ranks),
        'failures': failures}
