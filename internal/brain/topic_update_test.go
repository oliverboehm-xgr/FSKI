package brain

import "testing"

func TestUpdateActiveTopic_KeepsTopicOnGenericShortFollowup(t *testing.T) {
	ws := &Workspace{ActiveTopic: "nachrichten"}

	got := UpdateActiveTopic(ws, "lass uns darüber sprechen")
	if got != "nachrichten" {
		t.Fatalf("expected topic to stay 'nachrichten', got %q", got)
	}
}
