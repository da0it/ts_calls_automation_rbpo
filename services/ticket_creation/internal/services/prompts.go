package services

import "fmt"

func ticketSummarySystemPrompt() string {
	return `Верни только JSON без markdown и пояснений:
{"problem":"одно короткое предложение с сутью проблемы клиента"}
Правила: по-русски, не выдумывай факты, не повторяй intent, если данных мало — пустая строка, без лишних персональных данных.`
}

func buildSummaryUserPrompt(transcript, intentID, priority, entitiesInfo string) string {
	return fmt.Sprintf(`Транскрипция:
%s

Intent: %s
Приоритет: %s

Сущности:
%s

Верни только JSON.`, transcript, intentID, priority, entitiesInfo)
}
