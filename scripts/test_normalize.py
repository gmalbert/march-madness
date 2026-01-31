from predictions import normalize_team_name

# Test different Kansas State variations
test_names = [
    'Kansas State Wildcats',
    'Kansas St Wildcats',
    'Kansas State',
    'Kansas St'
]

for name in test_names:
    normalized = normalize_team_name(name)
    print(f'"{name}" -> "{normalized}"')