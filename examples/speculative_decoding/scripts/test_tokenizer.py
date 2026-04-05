# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Playground script to verify _process_chat_sample tokenization with answer_only_loss.

Green tokens = loss is computed (assistant), Red tokens = loss is masked (user/system/special).

Usage:
    python scripts/test_tokenizer.py
"""

import sys

sys.path.insert(0, ".")

import transformers

from modelopt.torch.utils.plugins.transformers_dataset import LanguageDataCollator, _sharegpt_to_openai_messages

MODEL = "openai/gpt-oss-120b"

# EXAMPLE_CONVERSATION = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"},
#     {"role": "assistant", "content": "The capital of France is Paris.", "thinking": "I'm using my knowledge of world capitals to answer the user's question about France's capital."},
#     {"role": "user", "content": "And what about Germany?"},
#     {"role": "assistant", "content": "The capital of Germany is Berlin.", "thinking": "I'm using my knowledge of world capitals to answer the user's question about Germany's capital."},
# ]

EXAMPLE_CONVERSATION = [
{
"role": "system",
"content": "You are a customer service agent that helps the user. The policy that determines how you should respond to requests from users is described below between the <policy> and </policy> tags.\n\nIn each turn you can either:\n- Send a message to the user.\n- Make a tool call.\nYou cannot do both at the same time.\n\n<policy>\n**Food Delivery Agent Policy** \n*The current time is 2024-06-01 10:00:00 EST.* \n\n---\n\n### **Core Rules** \n1. **Authentication Required**: \n - Always verify user identity via ID/phone/email before account access \n - Guest users must provide order confirmation number + last 4 digits of payment method \n\n2. **Action Confirmation**: \n - Require explicit verbal confirmation for all order changes/cancellations \n - Document all confirmations with timestamps \n\n3. **Payment Security**: \n - Only use payment methods already in user profile \n - Maximum 2 payment methods per order: \n - Primary: Digital wallet (mandatory) \n - Secondary: Gift card or loyalty points (optional) \n - Loyalty points may cover maximum 30% of order total \n\n4. **System Interaction**: \n - Make one tool call at a time (no parallel actions) \n - Validate restaurant availability before order placement \n\n---\n\n### **Domain Basics** \n\n**User Profile** \n- User ID (required for account holders) \n- Contact information (phone/email) \n- Payment methods: Credit/Debit, Digital Wallet, Gift Cards \n- Loyalty status: Regular (0-49 pts), Silver (50-199), Gold (200+) \n- Dietary preference flags \n\n**Restaurant Partners** \n- Restaurant ID + health inspection rating \n- Real-time menu availability updates every 90 seconds \n- Delivery zones (max 5 mile radius) \n- Temperature control certifications \n\n**Order Components** \n- Order ID \n- Restaurant ID(s) \n- Items with customization notes \n- Delivery timeline: \n - Pending ➔ Preparing ➔ En Route ➔ Delivered \n - Status changes locked 2 minutes before estimated delivery \n\n---\n\n### **Order Handling** \n\n**1. Place Order** \n- Multi-restaurant orders: \n - Split into separate orders per restaurant \n - Apply separate delivery fees ($3.99 base + $0.50/mile beyond 3 miles) \n - Modifications to one restaurant's items don't affect others \n\n- Required checks: \n - Confirm delivery address within restaurant's zone \n - Flag common allergens (peanuts, shellfish, gluten) \n\n**2. Modify Order** \n- Time restrictions: \n - Menu changes allowed only within 8 minutes of ordering \n - Address changes prohibited after \"preparing\" status \n\n- Ingredient rules: \n - Removals: Always permitted \n - Additions: Require restaurant confirmation via check_ingredient_availability \n\n\n**3. Cancel Order** \n- Refund tiers: \n - Full refund: Within 15 minutes of placement \n - Partial refund: 50% during preparation phase \n - No refund: After \"en route\" status \n\n- Loyalty impact: \n - Deduct 2x earned points for cancelled orders \n - Max 100 bonus points revocable per cancellation \n\n---\n\n### **Delivery Protocols** \n\n**Attempt Sequence** \n1. First attempt: Standard delivery \n2. Second attempt (if requested): +$5 fee \n3. Final disposition: \n - Perishables donated after 2 hours \n - Non-perishables returned to restaurant \n\n**Verification Requirements** \n- Mandatory photo proof containing: \n - Order ID label \n - Thermal packaging for hot items (min 145°F) \n - Insulated cold packaging for refrigerated items \n- Users may claim $3 credit for missing temp proof \n\n---\n\n### **Refunds & Credits** \n\n**Automatic Eligibility** \n- Wrong/missing items (require photo within 15m of delivery) \n- >30min late delivery (excludes severe weather events) \n- Temperature violations (melted ice cream, cold pizza) \n\n**Non-Refundable Items** \n- Customized menu items \n- Perishables showing consumption evidence \n- Alcohol/tobacco products \n\n**Compensation** \n- Service failures: 1 loyalty credit = $5 \n- Delivery errors: Max 3 credits per order \n\n---\n\n### **Special Cases** \n\n**Guest Checkouts** \n- No modifications allowed \n- Max order value = $75 \n- Refunds only via platform credit \n- Mandatory ID verification for alcohol \n\n**Allergy Safeguards** \n- Required confirmation dialog: \n *\"You selected peanut-containing items. Confirm no peanut allergies: [Yes/No]\"* \n- Automatic cancellation if \"No\" selected + allergy warning \n\n**Weather Exceptions** \n- No late penalties during: \n - Snow emergencies \n - Heat advisories (>95°F) \n - Severe thunderstorms \n\n---\n\n### **Escalation Protocol** \nTransfer to human agent if: \n1. Special dietary needs require chef consultation \n2. Physical item inspection requested \n3. Three consecutive failed delivery attempts \n4. Alcohol/tobacco ID verification failures \n\n*Transfer process: First call transfer_to_human_agents tool, then send: \n\"YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.\"*\n\n</policy>\n\nTry to be helpful and always follow the policy.",
"tool_calls": None,
"reasoning_content": None,
"tool_call_id": None
},
{
"role": "user",
"content": "\n\nHello, I'm trying to update my order (ORD1058) with a new payment method. My user ID is SECURE1. When I attempt to add my new VISA 5555 card during the modification process, the system says it's not allowed. Can you clarify why I can't add a new card directly during an order change?",
"tool_calls": None,
"reasoning_content": None,
"tool_call_id": None
},
{
"role": "assistant",
"content": "",
"tool_calls": [
{
"type": "function",
"id": "chatcmpl-tool-877c1a2d5cb149f685397719015694ed",
"function": {
"name": "authenticate_user",
"arguments": "{\"user_id\": \"SECURE1\", \"auth_method\": \"ID\", \"verification_code\": \"123456\"}"
}
}
],
"reasoning_content": "Thinking about how to call this tool...",
"tool_call_id": None
},
{
"role": "tool",
"content": "\n\n{\"access_token\": \"AUTH_TOKEN_7890\", \"auth_status\": \"verified\"}",
"tool_calls": None,
"reasoning_content": None,
"tool_call_id": "chatcmpl-tool-877c1a2d5cb149f685397719015694ed"
},
{
"role": "assistant",
"content": "\n\nThe system is designed to only allow the use of payment methods that are already saved in your user profile for security and privacy reasons. If you want to add a new payment method like your VISA card ending in 5555, you would need to add it to your profile first. Once the card is added to your profile, you can use it for placing new orders or modifying existing ones.\n\nWould you like me to assist you with adding the new card to your profile so you can use it for this order modification?",
"tool_calls": None,
"reasoning_content": "Thinking about my response...",
"tool_call_id": None
}
]

tools = [
{
"type": "function",
"function": {
"name": "place_order",
"description": "Create a new food delivery order",
"parameters": {
"properties": {
"user_id": {
"type": "string"
},
"restaurants": {
"items": {
"properties": {
"restaurant_id": {
"type": "string"
},
"items": {
"items": {
"properties": {
"item_id": {
"type": "string"
},
"customizations": {
"type": "string"
}
},
"required": [
"item_id"
]
},
"type": "array"
}
},
"required": [
"restaurant_id",
"items"
],
"type": "object"
},
"type": "array"
},
"payment_methods": {
"items": {
"properties": {
"payment_id": {
"type": "string"
},
"amount": {
"type": "number"
}
},
"required": [
"payment_id",
"amount"
]
},
"type": "array"
},
"delivery_address": {
"properties": {
"street": {
"type": "string"
},
"city": {
"type": "string"
},
"zip": {
"type": "string"
}
},
"required": [
"street",
"city",
"zip"
],
"type": "object"
},
"loyalty_points": {
"type": "integer"
},
"dietary_flags": {
"type": "array"
},
"allergen_warning_acknowledged": {
"type": "boolean"
},
"order_id": None,
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"user_id",
"restaurants",
"payment_methods",
"delivery_address",
"allergen_warning_acknowledged"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "modify_order",
"description": "Modify an existing order within allowed time window",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": {
"type": "string"
},
"modifications": {
"properties": {
"items_add": {
"type": "array"
},
"items_remove": {
"type": "array"
},
"address_change": {
"type": "object"
}
},
"type": "object"
},
"modification_timestamp": {
"type": "string"
},
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"order_id",
"modifications",
"modification_timestamp"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "cancel_order",
"description": "Cancel an order and process refund based on policy.",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": {
"type": "string"
},
"modifications": None,
"modification_timestamp": None,
"reason": {
"enum": [
"user_request",
"restaurant_closed",
"delivery_issue"
]
},
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"order_id",
"reason"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "get_order_status",
"description": "Retrieve current status and details of an order.",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": {
"type": "string"
},
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"order_id"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "check_ingredient_availability",
"description": "Verify if restaurant can accommodate custom ingredient requests.",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": None,
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": {
"type": "string"
},
"item_id": {
"type": "string"
},
"requested_mods": {
"type": "array"
},
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"restaurant_id",
"item_id"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "get_restaurant_menu",
"description": "Retrieve real-time menu for a restaurant",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": None,
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": {
"type": "string"
},
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"restaurant_id"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "verify_guest_order",
"description": "Authenticate guest order access.",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": None,
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": {
"type": "string"
},
"payment_digits": {
"type": "string"
},
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"confirmation_number",
"payment_digits"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "process_refund",
"description": "Handle refunds according to policy rules",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": {
"type": "string"
},
"modifications": None,
"modification_timestamp": None,
"reason": {
"enum": [
"missing_items",
"temperature_violation",
"late_delivery"
]
},
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": {
"type": "string"
},
"refund_tier": {
"enum": [
"full",
"partial",
"none"
]
},
"items_ineligible": {
"type": "array"
},
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"order_id",
"reason",
"refund_tier"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "update_delivery_status",
"description": "Record delivery attempts and status changes.",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": {
"type": "string"
},
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": {
"type": "integer"
},
"proof_photos": {
"type": "array"
},
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"order_id",
"attempt_number"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "transfer_to_human_agents",
"description": "Escalate complex issues to human support.",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": None,
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": {
"type": "string"
},
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"issue_summary"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "authenticate_user",
"description": "Verify user identity through specified authentication method",
"parameters": {
"properties": {
"user_id": {
"type": "string"
},
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": None,
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": {
"enum": [
"ID",
"phone",
"email"
]
},
"verification_code": {
"type": "string"
},
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"user_id",
"auth_method",
"verification_code"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "validate_payment_methods",
"description": "Ensure payment methods comply with policy requirements",
"parameters": {
"properties": {
"user_id": {
"type": "string"
},
"restaurants": None,
"payment_methods": {
"items": {
"properties": {
"payment_id": {
"type": "string"
},
"amount": {
"type": "number"
}
},
"required": [
"payment_id",
"amount"
]
},
"type": "array"
},
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": None,
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": {
"type": "number"
},
"allergen_acknowledged": None
},
"required": [
"user_id",
"order_total",
"payment_methods"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "handle_delivery_exceptions",
"description": "Manage delivery attempts and final disposition",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": {
"type": "string"
},
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": {
"type": "integer"
},
"proof_photos": {
"type": "array"
},
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": None
},
"required": [
"order_id",
"attempt_number"
]
},
"strict": True
}
},
{
"type": "function",
"function": {
"name": "allergy_confirmation",
"description": "Handle mandatory allergy acknowledgment workflow",
"parameters": {
"properties": {
"user_id": None,
"restaurants": None,
"payment_methods": None,
"delivery_address": None,
"loyalty_points": None,
"dietary_flags": None,
"allergen_warning_acknowledged": None,
"order_id": {
"type": "string"
},
"modifications": None,
"modification_timestamp": None,
"reason": None,
"restaurant_id": None,
"item_id": None,
"requested_mods": None,
"confirmation_number": None,
"payment_digits": None,
"evidence": None,
"refund_tier": None,
"items_ineligible": None,
"attempt_number": None,
"proof_photos": None,
"issue_summary": None,
"auth_method": None,
"verification_code": None,
"order_total": None,
"allergen_acknowledged": {
"type": "boolean"
}
},
"required": [
"order_id",
"allergen_acknowledged"
]
},
"strict": True
}
}
]

EXAMPLE_CONVERSATION = _sharegpt_to_openai_messages(EXAMPLE_CONVERSATION)

# Sanitize tools and conversation to remove None values
for tool in tools:
    func = tool.get("function", {})
    params = func.get("parameters", {})
    props = params.get("properties", {})
    for key in list(props.keys()):
        if props[key] is None:
            del props[key]
for msg in EXAMPLE_CONVERSATION:
    for key in list(msg.keys()):
        if msg[key] is None:
            del msg[key]

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def main():
    print(f"Loading tokenizer from {MODEL}...")
    with open("/root/eagle/ModelOptNew/gptoss_chat_template.jinja", "r") as f:
        template_content = f.read()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, chat_template=template_content)

    collator = LanguageDataCollator(
        tokenizer=tokenizer,
        train_len=16384,
        return_labels=True,
        answer_only_loss=True,
        bucket_granularity=0,
    )

    tokenized = collator._process_chat_sample([EXAMPLE_CONVERSATION], tools=tools)

    input_ids = tokenized["input_ids"].squeeze(0)
    labels = tokenized["labels"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)
    assistant_masks = tokenized.get("assistant_masks")

    real_len = attention_mask.sum().item()
    input_ids = input_ids[:real_len]
    labels = labels[:real_len]
    if assistant_masks is not None:
        if hasattr(assistant_masks, "squeeze"):
            assistant_masks = assistant_masks.squeeze(0)
        assistant_masks = assistant_masks[:real_len]

    print(f"\nSequence length: {real_len}")
    print(f"assistant_masks present: {assistant_masks is not None}")
    if assistant_masks is not None:
        num_active = sum(1 for m in assistant_masks if m)
        print(f"Loss tokens (green): {num_active}/{real_len}")

    # Detokenize per-token and colorize
    print(f"\n{'=' * 80}")
    print("Colorized output (GREEN=loss active, RED=loss masked):")
    print(f"{'=' * 80}\n")

    output = []
    for idx in range(real_len):
        tok_str = tokenizer.decode([input_ids[idx].item()])
        if assistant_masks is not None:
            is_active = bool(assistant_masks[idx])
        else:
            is_active = False
        color = GREEN if is_active else RED
        output.append(f"{color}{tok_str}{RESET}")

    print("".join(output))
    print(f"\n{'=' * 80}")

    # Also print raw token IDs for first/last few tokens
    print("\nFirst 20 tokens (id -> decoded -> mask):")
    for idx in range(min(20, real_len)):
        tok_str = tokenizer.decode([input_ids[idx].item()])
        mask_val = bool(assistant_masks[idx]) if assistant_masks is not None else "N/A"
        print(f"  [{idx:3d}] {input_ids[idx].item():6d} -> {tok_str!r:20s} mask={mask_val}")


if __name__ == "__main__":
    main()
