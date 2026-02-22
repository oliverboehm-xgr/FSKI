package brain

import "testing"

func TestUpdateActiveTopic_KeepsTopicOnFollowupReference(t *testing.T) {
	ws := &Workspace{ActiveTopic: "nachrichten"}

	got := UpdateActiveTopic(ws, "Lass uns über die Nachricht 4 sprechen")
	if got != "nachrichten" {
		t.Fatalf("expected topic to stay 'nachrichten', got %q", got)
	}
}
